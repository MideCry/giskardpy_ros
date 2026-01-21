from typing import Optional

from pkg_resources import resource_filename

from giskardpy.model.world_config import WorldWithFixedRobot
from semantic_digital_twin.robots.dualarm import Dualarm

from giskardpy_ros.configs.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import CollisionCheckingConfig


class WorldWithDualarmConfig(WorldWithFixedRobot):
    """Minimal Tracy world config analogous to WorldWithPR2Config.

    - Fixed-base robot (no drive joint)
    - Accepts URDF via argument; if not provided, reads from ROS parameter server
    - Applies conservative default motion limits
    """

    def __init__(self, urdf: Optional[str] = None):
        super().__init__(urdf=urdf, root_name=PrefixedName("map2"), urdf_view=Dualarm)

    def setup_collision_config(self):
        # TODO: create dualarm srdf
        path_to_srdf = resource_filename(
            "giskardpy", "../self_collision_matrices/iai/dualarm.srdf"
        )
        self.world.load_collision_srdf(path_to_srdf)

        for body in self.robot.bodies_with_collisions:
            collision_config = CollisionCheckingConfig(
                buffer_zone_distance=0.03, violated_distance=0.0
            )
            body.set_static_collision_config(collision_config)

    def setup_world(self, robot_name: Optional[str] = None) -> None:
        super().setup_world()
        self.robot = self.world.get_semantic_annotations_by_type(Dualarm)[0]


class DualarmStandAloneRobotInterfaceConfig(StandAloneRobotInterfaceConfig):
    def __init__(self):
        super().__init__(
            # TODO: change to dualarm config
            [
                "left_shoulder_pan_joint",
                "left_shoulder_lift_joint",
                "left_elbow_joint",
                "left_wrist_1_joint",
                "left_wrist_2_joint",
                "left_wrist_3_joint",
                "right_shoulder_pan_joint",
                "right_shoulder_lift_joint",
                "right_elbow_joint",
                "right_wrist_1_joint",
                "right_wrist_2_joint",
                "right_wrist_3_joint",
            ]
        )
