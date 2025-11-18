from dataclasses import dataclass
from typing import Optional

import numpy as np
from pkg_resources import resource_filename

from giskardpy.model.world_config import WorldConfig, WorldWithOmniDriveRobot
from giskardpy_ros.configs.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
    RobotInterfaceConfig,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world_description.connections import ActiveConnection, OmniDrive
from semantic_digital_twin.world_description.world_entity import CollisionCheckingConfig


@dataclass
class WorldWithHSRConfig(WorldWithOmniDriveRobot):

    def setup_world(self):
        super().setup_world()
        self.hsr = HSRB.from_world(self.world)

    def setup_collision_config(self):
        path_to_srdf = resource_filename(
            "giskardpy", "../self_collision_matrices/iai/hsrb.srdf"
        )
        self.world.load_collision_srdf(path_to_srdf)
        for body in self.hsr.bodies_with_collisions:
            collision_config = CollisionCheckingConfig(
                buffer_zone_distance=0.05, violated_distance=0.0
            )
            body.set_static_collision_config(collision_config)

        connection: ActiveConnection = self.world.get_connection_by_name(
            "wrist_roll_joint"
        )
        connection.set_static_collision_config_for_direct_child_bodies(
            CollisionCheckingConfig(
                buffer_zone_distance=0.05,
                violated_distance=0.0,
                max_avoided_bodies=4,
            )
        )

        connection: ActiveConnection = self.hsr.drive
        connection.set_static_collision_config_for_direct_child_bodies(
            CollisionCheckingConfig(
                buffer_zone_distance=0.1,
                violated_distance=0.03,
                max_avoided_bodies=2,
            )
        )

        connection: ActiveConnection = self.world.get_connection_by_name(
            "head_tilt_joint"
        )
        connection.set_static_collision_config_for_direct_child_bodies(
            CollisionCheckingConfig(
                buffer_zone_distance=0.03,
            )
        )
        # self.set_default_limits(
        #     {
        #         Derivatives.velocity: 1,
        #         Derivatives.acceleration: np.inf,
        #         Derivatives.jerk: None,
        #     }
        # )
        # self.add_empty_link(PrefixedName(self.map_name))
        # self.add_6dof_joint(
        #     parent_link=self.map_name,
        #     child_link=self.odom_link_name,
        #     joint_name=self.localization_joint_name,
        # )
        # self.add_empty_link(PrefixedName(self.odom_link_name))
        # self.add_robot_urdf(urdf=self.robot_description)
        # root_link_name = self.get_root_link_of_group(self.robot_group_name)
        # self.add_omni_drive_joint(
        #     parent_link_name=self.odom_link_name,
        #     child_link_name=root_link_name,
        #     name=self.drive_joint_name,
        #     x_name=PrefixedName("odom_x", self.robot_group_name),
        #     y_name=PrefixedName("odom_y", self.robot_group_name),
        #     yaw_vel_name=PrefixedName("odom_t", self.robot_group_name),
        #     translation_limits={
        #         Derivatives.velocity: 0.2,
        #         Derivatives.acceleration: np.inf,
        #         Derivatives.jerk: None,
        #     },
        #     rotation_limits={
        #         Derivatives.velocity: 0.2,
        #         Derivatives.acceleration: np.inf,
        #         Derivatives.jerk: None,
        #     },
        #     robot_group_name=self.robot_group_name,
        # )
        # self.set_joint_limits(
        #     limit_map={
        #         Derivatives.jerk: None,
        #     },
        #     joint_name="arm_lift_joint",
        # )


class HSRStandaloneInterface(RobotInterfaceConfig):
    def setup(self):
        self.register_controlled_joints(
            [
                "arm_flex_joint",
                "arm_lift_joint",
                "arm_roll_joint",
                "head_pan_joint",
                "head_tilt_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
                self.world.get_connections_by_type(OmniDrive)[0].name,
            ]
        )


class HSRVelocityInterface(RobotInterfaceConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(
        self,
        map_name: str = "map",
        localization_joint_name: str = "localization",
        odom_link_name: str = "odom",
        drive_joint_name: str = "brumbrum",
    ):
        self.map_name = map_name
        self.localization_joint_name = localization_joint_name
        self.odom_link_name = odom_link_name
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.sync_6dof_joint_with_tf_frame(
            joint_name=self.localization_joint_name,
            tf_parent_frame=self.map_name,
            tf_child_frame=self.odom_link_name,
        )
        self.sync_joint_state_topic("/hsrb/joint_states")
        self.sync_odometry_topic("/hsrb/odom", self.drive_joint_name)

        self.add_joint_velocity_group_controller(
            namespace="/hsrb/realtime_body_controller_real"
        )

        self.add_base_cmd_velocity(
            cmd_vel_topic="/hsrb/command_velocity", joint_name=self.drive_joint_name
        )


class HSRJointTrajInterfaceConfig(RobotInterfaceConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(
        self,
        map_name: str = "map",
        localization_joint_name: str = "localization",
        odom_link_name: str = "odom",
        drive_joint_name: str = "brumbrum",
    ):
        self.map_name = map_name
        self.localization_joint_name = localization_joint_name
        self.odom_link_name = odom_link_name
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.sync_6dof_joint_with_tf_frame(
            joint_name=self.localization_joint_name,
            tf_parent_frame=self.map_name,
            tf_child_frame=self.odom_link_name,
        )
        self.sync_joint_state_topic("/hsrb/joint_states")
        self.sync_odometry_topic("/hsrb/odom", self.drive_joint_name)

        self.add_follow_joint_trajectory_server(
            namespace="/hsrb/head_trajectory_controller", fill_velocity_values=True
        )
        self.add_follow_joint_trajectory_server(
            namespace="/hsrb/arm_trajectory_controller", fill_velocity_values=True
        )
        self.add_follow_joint_trajectory_server(
            namespace="/hsrb/omni_base_controller",
            fill_velocity_values=True,
            path_tolerance={
                Derivatives.position: 1,
                Derivatives.velocity: 1,
                Derivatives.acceleration: 100,
            },
        )
        # self.add_base_cmd_velocity(cmd_vel_topic='/hsrb/command_velocity',
        #                            track_only_velocity=True,
        #                            joint_name=self.drive_joint_name)


class HSRMujocoVelocityInterface(RobotInterfaceConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(
        self,
        map_name: str = "map",
        localization_joint_name: str = "localization",
        odom_link_name: str = "odom",
        drive_joint_name: str = "brumbrum",
    ):
        self.map_name = map_name
        self.localization_joint_name = localization_joint_name
        self.odom_link_name = odom_link_name
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.sync_6dof_joint_with_tf_frame(
            joint_name=self.localization_joint_name,
            tf_parent_frame=self.map_name,
            tf_child_frame=self.odom_link_name,
        )
        self.sync_joint_state_topic("/hsrb4s/joint_states")
        self.sync_odometry_topic("/hsrb4s/base_footprint", self.drive_joint_name)

        self.add_joint_velocity_controller(
            namespaces=[
                "hsrb4s/arm_flex_joint_velocity_controller",
                "hsrb4s/arm_lift_joint_velocity_controller",
                "hsrb4s/arm_roll_joint_velocity_controller",
                "hsrb4s/head_pan_joint_velocity_controller",
                "hsrb4s/head_tilt_joint_velocity_controller",
                "hsrb4s/wrist_flex_joint_velocity_controller",
                "hsrb4s/wrist_roll_joint_velocity_controller",
            ]
        )

        self.add_base_cmd_velocity(
            cmd_vel_topic="/hsrb4s/cmd_vel", joint_name=self.drive_joint_name
        )


class HSRMujocoPositionInterface(RobotInterfaceConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(
        self,
        map_name: str = "map",
        localization_joint_name: str = "localization",
        odom_link_name: str = "odom",
        drive_joint_name: str = "brumbrum",
    ):
        self.map_name = map_name
        self.localization_joint_name = localization_joint_name
        self.odom_link_name = odom_link_name
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.sync_6dof_joint_with_tf_frame(
            joint_name=self.localization_joint_name,
            tf_parent_frame=self.map_name,
            tf_child_frame=self.odom_link_name,
        )
        self.sync_joint_state_topic("/hsrb4s/joint_states")
        self.sync_odometry_topic("/hsrb4s/base_footprint", self.drive_joint_name)

        self.add_joint_position_controller(
            namespaces=[
                "hsrb4s/arm_flex_joint_position_controller",
                # 'hsrb4s/arm_lift_joint_position_controller',
                "hsrb4s/arm_roll_joint_position_controller",
                "hsrb4s/head_pan_joint_position_controller",
                "hsrb4s/head_tilt_joint_position_controller",
                "hsrb4s/wrist_flex_joint_position_controller",
                "hsrb4s/wrist_roll_joint_position_controller",
            ]
        )

        self.add_joint_velocity_controller(
            namespaces=[
                # 'hsrb4s/arm_flex_joint_position_controller',
                "hsrb4s/arm_lift_joint_position_controller",
                # 'hsrb4s/arm_roll_joint_position_controller',
                # 'hsrb4s/head_pan_joint_position_controller',
                # 'hsrb4s/head_tilt_joint_position_controller',
                # 'hsrb4s/wrist_flex_joint_position_controller',
                # 'hsrb4s/wrist_roll_joint_position_controller'
            ]
        )

        self.add_base_cmd_velocity(
            cmd_vel_topic="/hsrb4s/cmd_vel", joint_name=self.drive_joint_name
        )
