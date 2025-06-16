from copy import deepcopy
from typing import Optional

import numpy as np
from geometry_msgs.msg import Vector3Stamped
from rospy import Subscriber

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.data_types import ColorRGBA, PrefixName
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from giskardpy.motion_statechart.tasks.task import WEIGHT_BELOW_CA, Task
from giskardpy.symbol_manager import symbol_manager
from giskardpy_ros.ros1 import msg_converter


class HandleOffsetCorrection(Task):

    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 goal_vector: cas.Vector3,
                 threshold: float = 50,
                 name: Optional[str] = None,
                 weight: float = WEIGHT_BELOW_CA,
                 magic: float = 20):
        if name is None:
            name = f'{self.__class__.__name__}'
        super().__init__(name=name)
        self.root = root_link
        self.tip = tip_link
        self.threshold = threshold
        self.magic = magic

        self.root_V_goal_point = god_map.world.transform(self.root, goal_vector).to_np()

        root_V_goal_point: cas.Vector3 = symbol_manager.get_expr(self.ref_str +
                                                                 '.root_V_goal_point',
                                                                 input_type_hint=np.ndarray,
                                                                 output_type_hint=cas.Vector3)

        root_V_goal_point_not_normed = deepcopy(root_V_goal_point)

        root_V_goal_point.scale(1)
        root_V_goal_point.reference_frame = self.root
        root_V_goal_point.vis_frame = self.tip

        root_P_tip = god_map.world.compose_fk_expression(self.root, self.tip).to_position()

        root_P_goal_point = root_P_tip + root_V_goal_point

        god_map.debug_expression_manager.add_debug_expression('root_V_goal_point',
                                                              root_V_goal_point,
                                                              color=ColorRGBA(r=1, g=0, b=0, a=1))
        god_map.debug_expression_manager.add_debug_expression('root_P_tip',
                                                              root_P_tip,
                                                              color=ColorRGBA(r=0, g=0, b=1, a=1))
        god_map.debug_expression_manager.add_debug_expression('root_P_goal_point_normed',
                                                              root_P_goal_point,
                                                              color=ColorRGBA(r=0, g=1, b=0, a=1))

        root_V_error = (root_P_goal_point - root_P_tip) / self.magic
        root_V_stop = cas.Point3(reference_frame=root_P_goal_point.reference_frame)

        self.add_equality_constraint(reference_velocity=CartesianPosition.default_reference_velocity,
                                     equality_bound=root_V_stop.x,
                                     weight=weight,
                                     task_expression=root_P_tip.x,
                                     name=f'{name}/x')
        self.add_equality_constraint(reference_velocity=CartesianPosition.default_reference_velocity,
                                     equality_bound=root_V_error.y,
                                     weight=weight,
                                     task_expression=root_P_tip.y,
                                     name=f'{name}/y')
        self.add_equality_constraint(reference_velocity=CartesianPosition.default_reference_velocity,
                                     equality_bound=root_V_error.z,
                                     weight=weight,
                                     task_expression=root_P_tip.z,
                                     name=f'{name}/z')

        self.observation_expression = cas.less_equal(root_V_goal_point_not_normed.norm(), self.threshold)


class HandleOffsetCorrectionRealtime(HandleOffsetCorrection):

    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 threshold: float = 50,
                 name: Optional[str] = None,
                 weight: float = WEIGHT_BELOW_CA,
                 magic: float = 20):
        initial_vector = cas.Vector3().from_xyz(0, 0, 0,
                                                reference_frame=god_map.world.search_for_link_name('hand_camera_frame'))
        super().__init__(name=name,
                         root_link=root_link,
                         tip_link=tip_link,
                         threshold=threshold,
                         goal_vector=initial_vector,
                         weight=weight,
                         magic=magic)

        self.sub_offset = Subscriber(name='/robokudo/handle_offset', data_class=Vector3Stamped, callback=self.cb)

    def cb(self, data):
        data = msg_converter.ros_msg_to_giskard_obj(data, god_map.world)
        data = god_map.world.transform(self.root, data).to_np()
        self.root_V_goal_point = data
