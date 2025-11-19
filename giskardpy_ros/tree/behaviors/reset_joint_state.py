from py_trees.common import Status

from giskardpy.utils.decorators import record_time
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import (
    catch_and_raise_to_blackboard,
    GiskardBlackboard,
)


class ResetWorldState(GiskardBehavior):
    @record_time
    def __init__(self, name: str = "reset world state"):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        js = GiskardBlackboard().trajectory[0]
        GiskardBlackboard().executor.world.state = js
        GiskardBlackboard().executor.world.notify_state_change()
        return Status.SUCCESS
