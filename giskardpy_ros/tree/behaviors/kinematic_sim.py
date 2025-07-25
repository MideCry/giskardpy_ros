from line_profiler import profile
from py_trees.common import Status

from giskardpy.data_types.data_types import KeyDefaultDict
from giskardpy.god_map import god_map
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from line_profiler import profile


class KinSimPlugin(GiskardBehavior):

    @record_time
    def update(self):
        next_cmds = god_map.qp_solver_solution
        god_map.world.apply_control_commands(next_cmds, god_map.qp_controller.config.mpc_dt,
                                             derivative=god_map.qp_controller.config.max_derivative)
        # god_map.world.notify_state_change()
        return Status.SUCCESS
