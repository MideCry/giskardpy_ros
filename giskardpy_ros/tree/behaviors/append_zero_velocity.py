from py_trees.common import Status

from giskardpy.god_map import god_map
from giskardpy.utils.decorators import record_time
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from semantic_world.spatial_types.derivatives import Derivatives


class SetZeroVelocity(GiskardBehavior):

    def __init__(self, name=None):
        if name is None:
            name = 'set velocity to zero'
        super().__init__(name)

    @record_time

    def update(self):
        for free_variable, state in god_map.world.state.items():
            for derivative in Derivatives.range(Derivatives.velocity, Derivatives.jerk):
                if derivative == Derivatives.position:
                    continue
                god_map.world.state[free_variable][derivative] = 0
        god_map.world.notify_state_change()
        return Status.SUCCESS
