from line_profiler.explicit_profiler import profile
from py_trees.common import Status

from giskardpy.god_map import god_map
from giskardpy.qp.qp_controller import QPController
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils.decorators import record_time
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import catch_and_raise_to_blackboard


class ControllerPlugin(GiskardBehavior):
    controller: QPController = None
    controller2: QPController = None

    @catch_and_raise_to_blackboard
    def initialise(self):
        self.controller = god_map.qp_controller

    @catch_and_raise_to_blackboard(skip_on_exception=False)
    @record_time
    @profile
    def update(self):
        next_cmds = self.controller.get_cmd(symbol_manager)
        # if (self.controller2 is not None
        #         and next_cmds.check_base_variables_threshold(self.controller.free_variables)):
        #     parameters = self.controller2.get_parameter_names()
        #     substitutions = symbol_manager.resolve_symbols(parameters)
        #     next_cmds.free_variable_data.update(self.controller2.get_cmd(substitutions).free_variable_data)
        #     for v in self.controller.free_variables:
        #         if v.is_base:
        #             next_cmds.free_variable_data[v.name] = [0.,0.,0.]
        #             god_map.world.state[v.name].velocity = 0
        #             god_map.world.state[v.name].acceleration = 0
        #             god_map.world.state[v.name].jerk = 0

        god_map.qp_solver_solution = next_cmds
        return Status.RUNNING
