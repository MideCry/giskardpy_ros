from line_profiler.explicit_profiler import profile
from py_trees.common import Status

from giskardpy.god_map import god_map
from giskardpy.qp.qp_controller import QPController
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
        next_cmds = self.controller.get_cmd()
        god_map.qp_solver_solution = next_cmds
        return Status.RUNNING
