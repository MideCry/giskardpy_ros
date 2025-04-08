from copy import deepcopy

from line_profiler import profile
from py_trees.common import Status

from giskardpy.god_map import god_map
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from line_profiler import profile


class LogTrajPlugin(GiskardBehavior):
    @record_time
    @profile
    def update(self):
        god_map.trajectory.set(god_map.control_cycle_counter, god_map.world.state)
        return Status.SUCCESS
