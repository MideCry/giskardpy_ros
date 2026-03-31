from typing import Dict

import numpy as np
import pytest

from giskardpy.middleware.ros2 import rospy
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
    SetOdometry,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointState
from giskardpy.middleware.ros2.utils import load_xacro
from giskardpy.middleware.ros2.utils import GiskardTester
from krrood.symbolic_math.symbolic_math import trinary_logic_and
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
