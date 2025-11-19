from py_trees.common import Status

from giskardpy.utils.decorators import record_time
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import (
    catch_and_raise_to_blackboard,
    GiskardBlackboard,
)


class CollisionSceneUpdater(GiskardBehavior):
    def __init__(self):
        super().__init__("update collision scene")

    @catch_and_raise_to_blackboard(skip_on_exception=False)
    @record_time
    def update(self):
        GiskardBlackboard().executor.collision_scene.sync()
        return Status.SUCCESS
