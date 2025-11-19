from typing import Optional

from py_trees.common import Status

from giskardpy.utils.decorators import record_time
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import (
    catch_and_raise_to_blackboard,
    GiskardBlackboard,
)


class DebugMarkerPublisher(GiskardBehavior):
    def __init__(self, name: str = "debug marker"):
        super().__init__(name)
        self.debug_marker_visualizer = GiskardBlackboard().debug_marker_visualizer

    @record_time
    def update(self):
        GiskardBlackboard().debug_marker_visualizer.publish_markers(
            motion_statechart=GiskardBlackboard().motion_statechart
        )
        return Status.SUCCESS


class DebugMarkerPublisherTrajectory(GiskardBehavior):

    def __init__(self, name: Optional[str] = None, ensure_publish: bool = False):
        super().__init__(name)
        self.ensure_publish = ensure_publish
        self.every_x = 10

    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        return Status.FAILURE
