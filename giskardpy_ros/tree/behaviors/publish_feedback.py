import json
from typing import Optional

from giskard_msgs.action import JsonAction

import giskardpy_ros.ros2.msg_converter as msg_converter
import numpy as np
from giskard_msgs.msg import ExecutionState
from py_trees.common import Status
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from giskardpy.god_map import god_map
from giskardpy.utils.decorators import record_time
from giskardpy_ros.ros2 import rospy
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard


def did_state_change() -> bool:
    if len(god_map.motion_statechart_manager.task_state_history) <= 1:
        return False
    if len(god_map.motion_statechart_manager.task_state_history) == 2:
        return True
    last_task_state = god_map.motion_statechart_manager.task_state_history[-2][1][0]
    task_state = god_map.motion_statechart_manager.task_state_history[-1][1][0]
    if np.any(last_task_state != task_state):
        return True
    last_task_state = god_map.motion_statechart_manager.task_state_history[-2][1][1]
    task_state = god_map.motion_statechart_manager.task_state_history[-1][1][1]
    if np.any(last_task_state != task_state):
        return True
    last_monitor_state = god_map.motion_statechart_manager.monitor_state_history[-2][1][
        0
    ]
    monitor_state = god_map.motion_statechart_manager.monitor_state_history[-1][1][0]
    if np.any(last_monitor_state != monitor_state):
        return True
    last_monitor_state = god_map.motion_statechart_manager.monitor_state_history[-2][1][
        1
    ]
    monitor_state = god_map.motion_statechart_manager.monitor_state_history[-1][1][1]
    if np.any(last_monitor_state != monitor_state):
        return True
    last_goal_state = god_map.motion_statechart_manager.goal_state_history[-2][1][0]
    goal_state = god_map.motion_statechart_manager.goal_state_history[-1][1][0]
    if np.any(last_goal_state != goal_state):
        return True
    last_goal_state = god_map.motion_statechart_manager.goal_state_history[-2][1][1]
    goal_state = god_map.motion_statechart_manager.goal_state_history[-1][1][1]
    if np.any(last_goal_state != goal_state):
        return True
    return False


class PublishFeedback(GiskardBehavior):

    def __init__(self, name: Optional[str] = None, topic_name: Optional[str] = None):
        if name is None:
            name = self.__class__.__name__
        if topic_name is None:
            topic_name = f"{rospy.node.get_name()}/state"
        super().__init__(name)
        self.cmd_topic = topic_name
        self.move_action_server = GiskardBlackboard().move_action_server
        self.pub = rospy.node.create_publisher(
            ExecutionState,
            self.cmd_topic,
            QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.last_goal_id = -1

    @record_time
    def update(self):
        # FIXME
        # if did_state_change():
        data = {
            "life_cycle_state": GiskardBlackboard().motion_statechart.life_cycle_state.to_json(),
            "observation_state": GiskardBlackboard().motion_statechart.observation_state.to_json(),
        }
        if self.last_goal_id != self.move_action_server.goal_id:
            self.last_goal_id = self.move_action_server.goal_id
            data["motion_statechart"] = GiskardBlackboard().motion_statechart.to_json()
        data["goal_id"] = self.last_goal_id

        msg = JsonAction.Feedback()
        msg.feedback = json.dumps(data)
        self.move_action_server.send_feedback(msg)
        return Status.SUCCESS
