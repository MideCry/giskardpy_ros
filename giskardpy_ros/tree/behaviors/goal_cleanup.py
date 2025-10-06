from py_trees.common import Status

from giskardpy.god_map import god_map
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior


class GoalCleanUp(GiskardBehavior):
    def update(self):
        god_map.model_synchronizer.resume()
        god_map.state_synchronizer.resume()
        # fixme
        # for goal in god_map.motion_statechart_manager.motion_goals.values():
        #     goal.clean_up()
        return Status.SUCCESS
