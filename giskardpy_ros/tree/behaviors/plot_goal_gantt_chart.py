import traceback
from copy import copy
from typing import List, Dict, Tuple

import numpy as np
from py_trees.common import Status

from giskardpy.middleware import get_middleware
from giskardpy.motion_statechart.goals.collision_avoidance import CollisionAvoidance
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.utils.decorators import record_time
from giskardpy.utils.utils import create_path, cm_to_inch
from giskardpy_ros.tree.behaviors import plot_motion_graph
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard


class PlotGanttChart(GiskardBehavior):

    def __init__(self, name: str = "plot gantt chart"):
        super().__init__(name)

    @record_time
    def update(self):
        if not GiskardBlackboard().motion_statechart.history:
            return Status.SUCCESS
        try:
            file_name = (
                GiskardBlackboard().executor.tmp_folder
                + f"gantt_charts/goal_{GiskardBlackboard().move_action_server.goal_id}.pdf"
            )
            GiskardBlackboard().motion_statechart.plot_gantt_chart(
                file_name, context=GiskardBlackboard().executor.build_context
            )
        except Exception as e:
            get_middleware().logwarn(f"Failed to create goal gantt chart: {e}.")
            traceback.print_exc()

        return Status.SUCCESS
