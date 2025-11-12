from itertools import chain
from typing import List

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import EmptyProblemException
from giskardpy.god_map import god_map
from giskardpy.qp.constraint import (
    EqualityConstraint,
    InequalityConstraint,
    DerivativeInequalityConstraint,
    DerivativeEqualityConstraint,
)
from giskardpy.utils.decorators import record_time
from py_trees.common import Status

from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import (
    catch_and_raise_to_blackboard,
    GiskardBlackboard,
)


class CompileMotionStatechart(GiskardBehavior):
    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        GiskardBlackboard().executor.compile()
        return Status.SUCCESS
