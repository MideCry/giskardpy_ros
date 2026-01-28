from py_trees.common import Status

from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import (
    GiskardBlackboard,
    catch_and_raise_to_blackboard,
)


class PublishWorldState(GiskardBehavior):

    @catch_and_raise_to_blackboard
    def update(self):
        GiskardBlackboard().giskard.state_synchronizer.world_callback()
        return Status.SUCCESS
