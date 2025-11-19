from typing import Optional

from nav_msgs.msg import Odometry
from py_trees.common import Status

from giskardpy.middleware import get_middleware
from giskardpy.utils.decorators import record_time
from giskardpy_ros.ros2 import rospy, msg_converter
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import (
    catch_and_raise_to_blackboard,
    GiskardBlackboard,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName


class SyncOdometry(GiskardBehavior):

    def __init__(
        self,
        odometry_topic: str,
        joint_name: Optional[PrefixedName] = None,
        name_suffix: str = "",
    ):
        self.odometry_topic = odometry_topic
        if not self.odometry_topic.startswith("/"):
            self.odometry_topic = "/" + self.odometry_topic
        super().__init__(str(self) + name_suffix)
        self.joint = GiskardBlackboard().executor.world.get_drive_joint(
            joint_name=joint_name
        )
        self.odometry_sub = rospy.node.create_subscription(
            Odometry, self.odometry_topic, self.cb, 1
        )
        get_middleware().loginfo(f"Subscribed to {self.odometry_topic}")

    def __str__(self):
        return f"{super().__str__()} ({self.odometry_topic})"

    def cb(self, data: Odometry):
        self.odom = data

    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        trans_matrix = msg_converter.ros_msg_to_giskard_obj(
            self.odom.pose.pose, GiskardBlackboard().executor.world
        )
        self.joint.update_transform(trans_matrix)
        return Status.SUCCESS
