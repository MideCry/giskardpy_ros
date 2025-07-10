from typing import Optional

import numpy as np
import rospy
from geometry_msgs.msg import Vector3Stamped

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.data_types import PrefixName, ColorRGBA, Derivatives
from giskardpy.symbol_manager import symbol_manager
from giskardpy.motion_statechart.tasks.task import WEIGHT_BELOW_CA, Task
from giskardpy.god_map import god_map
from giskardpy_ros.ros1 import msg_converter


class VFHMoveDir(Task):
    """
    The VFH MoveDir task
    """

    def __init__(self,
                 tip_link: PrefixName,
                 goal_vector: cas.Vector3,
                 root_link: PrefixName,
                 max_velocity: float = 0.3,
                 weight: float = WEIGHT_BELOW_CA,
                 name: Optional[str] = None):
        self.tip_link = tip_link
        self.root = root_link
        self.max_velocity = max_velocity
        self.weight = weight

        if name is None:
            name = f'{self.__class__.__name__}/{self.root}/{self.tip_link}'
        super().__init__(name=name)

        self.root_V_goal_angle = god_map.world.transform(self.root, goal_vector).to_np()

        root_V_goal_angle: cas.Vector3 = symbol_manager.get_expr(self.ref_str +
                                                                 '.root_V_goal_angle',
                                                                 input_type_hint=np.ndarray,
                                                                 output_type_hint=cas.Vector3)
        root_V_goal_angle.vis_frame = tip_link

        root_V_goal_angle.reference_frame = self.root
        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip_link)
        root_P_tip = root_T_tip.to_position()
        root_P_goal_point = root_P_tip + root_V_goal_angle

        god_map.debug_expression_manager.add_debug_expression(name="root_P_goal_point",
                                                              expression=root_P_goal_point,
                                                              color=ColorRGBA(1.0, 1.0, 1.0, 1.0))
        god_map.debug_expression_manager.add_debug_expression(name="root_V_goal_angle",
                                                              expression=root_V_goal_angle,
                                                              color=ColorRGBA(0.0, 1.0, 1.0, 1.0))

        # maybe equality constraint instead?
        self.add_point_goal_constraints(frame_P_current=root_P_tip, frame_P_goal=root_P_goal_point,
                                        reference_velocity=self.max_velocity, weight=self.weight)


class RealMoveDir(VFHMoveDir):
    def __init__(self,
                 tip_link: PrefixName,
                 root_link: PrefixName,
                 topic_name: str,
                 max_velocity: float = 0.3,
                 weight: float = WEIGHT_BELOW_CA,
                 name: Optional[str] = None):
        initial_goal = cas.Vector3((0, 0, 0), reference_frame=god_map.world.search_for_link_name(tip_link))
        super().__init__(name=name,
                         tip_link=tip_link,
                         goal_vector=initial_goal,
                         root_link=root_link,
                         max_velocity=max_velocity,
                         weight=weight)
        self.sub = rospy.Subscriber(topic_name, Vector3Stamped, self.cb, queue_size=10)

    def cb(self, data: Vector3Stamped):
        data = msg_converter.ros_msg_to_giskard_obj(data, god_map.world)
        data = god_map.world.transform(self.root, data).to_np()
        self.root_V_goal_angle = data
