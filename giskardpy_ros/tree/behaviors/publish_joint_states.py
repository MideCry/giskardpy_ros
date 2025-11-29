from typing import Optional

from py_trees.common import Status
from sensor_msgs.msg import JointState

from giskardpy_ros.ros2 import rospy
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import (
    GiskardBlackboard,
    catch_and_raise_to_blackboard,
)


class PublishWorldState(GiskardBehavior):

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        if name is None:
            name = self.__class__.__name__
        super().__init__(name)

    @catch_and_raise_to_blackboard
    def update(self):
        GiskardBlackboard().giskard.state_synchronizer.world_callback()
        return Status.SUCCESS
