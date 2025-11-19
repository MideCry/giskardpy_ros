import json

from giskard_msgs.action import JsonAction
from py_trees.common import Status

import giskardpy_ros.ros2.msg_converter as msg_converter
from giskardpy.data_types.exceptions import *
from giskardpy.middleware import get_middleware
from giskardpy.utils.decorators import record_time
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard


class SetMoveResult(GiskardBehavior):

    def __init__(self, name, context: str, print=True):
        self.print = print
        self.context = context
        super().__init__(name)

    @record_time
    def update(self):
        e = self.get_blackboard_exception()
        if e is None:
            move_result = JsonAction.Result()
            GiskardBlackboard().move_action_server.set_succeeded()
        else:
            move_result = JsonAction.Result()
            GiskardBlackboard().move_action_server.set_aborted()

        # trajectory = god_map.trajectory
        # joints = GiskardBlackboard().executor.world.get_connections_by_type(ActiveConnection)
        # move_result.trajectory = msg_converter.trajectory_to_ros_trajectory(
        #     trajectory,
        #     sample_period=GiskardBlackboard().giskard.qp_controller_config.mpc_dt,
        #     start_time=0,
        #     joints=joints,
        # )

        result = {
            "life_cycle_state": GiskardBlackboard().motion_statechart.life_cycle_state.to_json(),
            "observation_state": GiskardBlackboard().motion_statechart.observation_state.to_json(),
        }

        move_result.result = json.dumps(result)
        if isinstance(e, PreemptedException):
            get_middleware().logwarn(f"Goal preempted.")
        else:
            if self.print:
                get_middleware().loginfo(f"{self.context} succeeded.")

        # move_result.execution_state = giskard_state_to_execution_state()
        GiskardBlackboard().move_action_server.result_msg = move_result
        # move_result.execution_state = giskard_state_to_execution_state()
        return Status.SUCCESS
