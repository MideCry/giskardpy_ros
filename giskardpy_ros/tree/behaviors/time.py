from typing import Optional

from line_profiler import profile
from py_trees.common import Status

from giskardpy.god_map import god_map
from giskardpy_ros.ros2 import rospy
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from line_profiler import profile


class TimePlugin(GiskardBehavior):
    def __init__(self, name: Optional[str] = 'time'):
        super().__init__(name)

    def update(self):
        god_map.time += god_map.qp_controller.config.mpc_dt
        return Status.SUCCESS


class ControlCycleCounter(GiskardBehavior):

    def __init__(self, name: Optional[str] = 'control cycle counter'):
        super().__init__(name)

    def update(self):
        god_map.control_cycle_counter += 1
        return Status.SUCCESS


class RosTime(GiskardBehavior):
    def __init__(self, name: Optional[str] = 'ros time'):
        super().__init__(name)

    @property
    def start_time(self) -> float:
        return god_map.motion_start_time

    def update(self):
        god_map.time = rospy.node.get_clock().now().nanoseconds / 1e9 - self.start_time
        return Status.SUCCESS
