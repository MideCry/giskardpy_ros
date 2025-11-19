from copy import deepcopy

from py_trees.common import Status

from giskardpy.model.trajectory import Trajectory
from giskardpy.utils.decorators import record_time
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard


class NewTrajectory(GiskardBehavior):
    @record_time
    def initialise(self):
        current_js = deepcopy(GiskardBlackboard().executor.world.state)
        trajectory = Trajectory()
        trajectory.append(current_js)
        GiskardBlackboard().trajectory = trajectory

    def update(self):
        return Status.SUCCESS
