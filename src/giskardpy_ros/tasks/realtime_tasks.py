from __future__ import division

from typing import Optional

import rospy
from geometry_msgs.msg import PointStamped

from giskardpy.data_types.data_types import PrefixName
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.tasks.task import WEIGHT_BELOW_CA
from giskardpy.motion_statechart.tasks.pointing import Pointing, PointingCone
import giskardpy.casadi_wrapper as cas
import giskardpy_ros.ros1.msg_converter as msg_converter


class RealTimePointing(Pointing):

    def __init__(self,
                 tip_link: PrefixName,
                 root_link: PrefixName,
                 topic_name: str,
                 pointing_axis: cas.Vector3,
                 name: Optional[str] = None,
                 max_velocity: float = 0.3,
                 weight: float = WEIGHT_BELOW_CA,
                 start_condition: cas.Expression = cas.BinaryTrue,
                 pause_condition: cas.Expression = cas.BinaryFalse,
                 end_condition: cas.Expression = cas.BinaryFalse):
        initial_goal = cas.Point3((1, 0, 1), reference_frame=god_map.world.search_for_link_name('base_footprint'))
        super().__init__(name=name,
                         tip_link=tip_link,
                         goal_point=initial_goal,
                         root_link=root_link,
                         pointing_axis=pointing_axis,
                         max_velocity=max_velocity,
                         weight=weight)
        self.sub = rospy.Subscriber(topic_name, PointStamped, self.cb)

    def cb(self, data: PointStamped):
        data = msg_converter.ros_msg_to_giskard_obj(data, god_map.world)
        data = god_map.world.transform(self.root, data).to_np()
        self.root_P_goal_point = data


class RealTimeConePointing(PointingCone):
    def __init__(self,
                 tip_link: PrefixName,
                 root_link: PrefixName,
                 topic_name: str,
                 pointing_axis: cas.Vector3,
                 cone_theta: float = 0.0,
                 name: Optional[str] = None,
                 max_velocity: float = 0.3,
                 threshold: float = 0.01,
                 weight: float = WEIGHT_BELOW_CA):
        initial_goal = cas.Point3((1, 0, 1), reference_frame=god_map.world.search_for_link_name('base_footprint'))
        super().__init__(name=name,
                         tip_link=tip_link,
                         goal_point=initial_goal,
                         root_link=root_link,
                         pointing_axis=pointing_axis,
                         cone_theta=cone_theta,
                         max_velocity=max_velocity,
                         threshold=threshold,
                         weight=weight)
        self.sub = rospy.Subscriber(topic_name, PointStamped, self.cb)

    def cb(self, data: PointStamped):
        data = msg_converter.ros_msg_to_giskard_obj(data, god_map.world)
        data = god_map.world.transform(self.root, data).to_np()
        self.root_P_goal_point = data
