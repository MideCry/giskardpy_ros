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
        move_result = JsonAction.Result()
        match e:
            case ExecutionCanceledException():
                GiskardBlackboard().move_action_server.set_canceled()
            case ExecutionAbortedException():
                GiskardBlackboard().move_action_server.set_aborted()
            case None:
                GiskardBlackboard().move_action_server.set_succeeded()

        result = {
            "life_cycle_state": GiskardBlackboard().motion_statechart.life_cycle_state.to_json(),
            "observation_state": GiskardBlackboard().motion_statechart.observation_state.to_json(),
        }

        move_result.result = json.dumps(result)
        if isinstance(e, ExecutionCanceledException):
            get_middleware().logwarn(f"Goal canceled by user.")
        else:
            if self.print:
                get_middleware().loginfo(f"{self.context} succeeded.")

        GiskardBlackboard().move_action_server.result_msg = move_result
        return Status.SUCCESS
