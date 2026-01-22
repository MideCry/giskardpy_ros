from enum import Enum

from py_trees.common import Status
from tf2_msgs.msg import TFMessage

from giskardpy_ros.ros2 import rospy
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import (
    GiskardBlackboard,
    catch_and_raise_to_blackboard,
)


class TfPublishingModes(Enum):
    nothing = 0
    all = 1
    attached_objects = 2

    world_objects = 4
    attached_and_world_objects = 6


class TFPublisher(GiskardBehavior):
    """
    Published tf for attached and environment objects.
    """

    def __init__(
        self,
        name: str,
        mode: TfPublishingModes,
        tf_topic: str = "tf",
        include_prefix: bool = True,
    ):
        super().__init__(name)
        self.original_links = set(
            body.name for body in GiskardBlackboard().executor.world.bodies
        )
        self.tf_pub = rospy.node.create_publisher(TFMessage, tf_topic, 10)
        self.mode = mode
        self.robots = GiskardBlackboard().giskard.robots
        self.include_prefix = include_prefix

    @catch_and_raise_to_blackboard
    def update(self):
        GiskardBlackboard().giskard.tf_publisher._notify()
        return Status.SUCCESS
