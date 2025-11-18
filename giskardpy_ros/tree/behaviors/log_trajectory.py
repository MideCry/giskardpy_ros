from py_trees.common import Status

from giskardpy.god_map import god_map
from giskardpy.utils.decorators import record_time
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior


class LogTrajPlugin(GiskardBehavior):
    @record_time
    def update(self):
        god_map.trajectory.append(god_map.world.state)
        return Status.SUCCESS
