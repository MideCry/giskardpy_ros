import typing
from typing import List, Optional

import numpy as np
from geometry_msgs.msg import Quaternion, Point
from line_profiler import profile
from py_trees.common import Status
from std_msgs.msg import ColorRGBA
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

import giskardpy.casadi_wrapper as w
from giskardpy.god_map import god_map
from giskardpy_ros.ros2 import rospy
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from giskardpy.utils.math import rotation_matrix_from_axis_angle, quaternion_from_rotation_matrix
from giskardpy_ros.tree.blackboard_utils import catch_and_raise_to_blackboard, GiskardBlackboard


class DebugMarkerPublisher(GiskardBehavior):
    colors = [ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),  # red
              ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),  # blue
              ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),  # yellow
              ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0),  # violet
              ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),  # cyan
              ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),  # green
              ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),  # white
              ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0),  # black
              ]

    @profile
    def __init__(self, name: str = 'debug marker', tf_topic: str = '/tf', map_frame: Optional[str] = None):
        super().__init__(name)
        if map_frame is None:
            self.map_frame = str(god_map.world.root_link_name)
        else:
            self.map_frame = map_frame
        self.tf_pub = rospy.node.create_publisher(TFMessage, tf_topic, 10)
        self.marker_pub = rospy.node.create_publisher(MarkerArray, f'{rospy.node.get_name()}/visualization_marker_array', 10)

    def setup(self, **kwargs: typing.Any) -> None:
        self.clear_markers()
        super().setup(**kwargs)

    def clear_markers(self):
        msg = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        msg.markers.append(marker)
        self.marker_pub.publish(msg)

    @record_time
    @profile
    def update(self):
        debug_exprs = god_map.debug_expression_manager.debug_expressions
        if len(debug_exprs) > 0:
            debug_state = god_map.debug_expression_manager.evaluated_debug_expressions
            ms = MarkerArray()
            markers = GiskardBlackboard().ros_visualizer.debug_state_to_vectors_markers(debug_exprs, debug_state)
            ms.markers.extend(markers)
            self.marker_pub.publish(ms)
        return Status.SUCCESS


class DebugMarkerPublisherTrajectory(GiskardBehavior):
    @profile
    def __init__(self,
                 name: Optional[str] = None,
                 ensure_publish: bool = False):
        super().__init__(name)
        self.ensure_publish = ensure_publish
        self.every_x = 10

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        debug_exprs = god_map.debug_expression_manager.debug_expressions
        if len(debug_exprs) > 0:
            debug_traj = god_map.debug_expression_manager._raw_debug_trajectory
            GiskardBlackboard().ros_visualizer.publish_debug_trajectory(debug_expressions=debug_exprs,
                                                                        raw_debug_trajectory=debug_traj,
                                                                        joint_space_traj=god_map.trajectory,
                                                                        every_x=self.every_x)
        return Status.SUCCESS
