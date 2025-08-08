from dataclasses import dataclass

from giskardpy.god_map import god_map
from giskardpy.model.collision_world_syncer import CollisionCheckerLib
from giskardpy.model.world_config import WorldWithOmniDriveRobot
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy_ros.configs.giskard import RobotInterfaceConfig
from semantic_world.connections import OmniDrive, RevoluteConnection
from semantic_world.prefixed_name import PrefixedName
from semantic_world.robots import PR2, CollisionAvoidanceThreshold

from pkg_resources import resource_filename


@dataclass
class WorldWithPR2Config(WorldWithOmniDriveRobot):
    odom_body_name: PrefixedName = PrefixedName('odom_combined')

    def setup(self):
        super().setup()
        pr2 = PR2.from_world(world=self.world)
        path_to_srdf = resource_filename('giskardpy', '../self_collision_matrices/iai/pr2.srdf')
        pr2.load_collision_config(path_to_srdf)
        pr2.collision_config.frozen_connections = {
            self.world.get_connection_by_name('r_gripper_l_finger_joint'),
            self.world.get_connection_by_name('l_gripper_l_finger_joint')
        }
        pr2.collision_config.default_external_threshold = CollisionAvoidanceThreshold(soft_threshold=0.1,
                                                                                      hard_threshold=0.0)
        for joint_name in ['r_wrist_roll_joint', 'l_wrist_roll_joint']:
            connection = self.world.get_connection_by_name(joint_name)
            threshold = CollisionAvoidanceThreshold(soft_threshold=0.05, hard_threshold=0.0,
                                                    number_of_repeller=4)
            pr2.collision_config.set_external_threshold_for_connection(connection=connection,
                                                                       threshold=threshold)

        for joint_name in ['r_wrist_flex_joint', 'l_wrist_flex_joint']:
            connection = self.world.get_connection_by_name(joint_name)
            threshold = CollisionAvoidanceThreshold(soft_threshold=0.05, hard_threshold=0.0,
                                                    number_of_repeller=2)
            pr2.collision_config.set_external_threshold_for_connection(connection=connection,
                                                                       threshold=threshold)
        for joint_name in ['r_elbow_flex_joint', 'l_elbow_flex_joint']:
            connection = self.world.get_connection_by_name(joint_name)
            threshold = CollisionAvoidanceThreshold(soft_threshold=0.05, hard_threshold=0.0)
            pr2.collision_config.set_external_threshold_for_connection(connection=connection,
                                                                       threshold=threshold)
        for joint_name in ['r_forearm_roll_joint', 'l_forearm_roll_joint']:
            connection = self.world.get_connection_by_name(joint_name)
            threshold = CollisionAvoidanceThreshold(soft_threshold=0.025, hard_threshold=0.0)
            pr2.collision_config.set_external_threshold_for_connection(connection=connection,
                                                                       threshold=threshold)
        drive_threshold = CollisionAvoidanceThreshold(soft_threshold=0.2, hard_threshold=0.1, number_of_repeller=2)
        pr2.collision_config.set_external_threshold_for_connection(connection=pr2.drive,
                                                                   threshold=drive_threshold)


class PR2StandaloneInterface(RobotInterfaceConfig):
    def setup(self):
        self.register_controlled_joints([
            'torso_lift_joint',
            'head_pan_joint',
            'head_tilt_joint',
            'r_shoulder_pan_joint',
            'r_shoulder_lift_joint',
            'r_upper_arm_roll_joint',
            'r_forearm_roll_joint',
            'r_elbow_flex_joint',
            'r_wrist_flex_joint',
            'r_wrist_roll_joint',
            'l_shoulder_pan_joint',
            'l_shoulder_lift_joint',
            'l_upper_arm_roll_joint',
            'l_forearm_roll_joint',
            'l_elbow_flex_joint',
            'l_wrist_flex_joint',
            'l_wrist_roll_joint',
            self.world.search_for_connections_of_type(OmniDrive)[0].name,
        ])


class PR2JointTrajServerMujocoInterface(RobotInterfaceConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(self,
                 map_name: str = 'map',
                 localization_joint_name: str = 'localization',
                 odom_link_name: str = 'odom_combined',
                 drive_joint_name: str = 'brumbrum'):
        self.map_name = map_name
        self.localization_joint_name = localization_joint_name
        self.odom_link_name = odom_link_name
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.sync_6dof_joint_with_tf_frame(joint_name=self.localization_joint_name,
                                           tf_parent_frame=self.map_name,
                                           tf_child_frame=self.odom_link_name)
        self.sync_joint_state_topic('/joint_states')
        self.sync_odometry_topic('/pr2/base_footprint', self.drive_joint_name)
        self.add_follow_joint_trajectory_server(
            namespace='/pr2/whole_body_controller')
        self.add_follow_joint_trajectory_server(
            namespace='/pr2/l_gripper_l_finger_controller')
        self.add_follow_joint_trajectory_server(
            namespace='/pr2/r_gripper_l_finger_controller')
        self.add_base_cmd_velocity(cmd_vel_topic='/pr2/cmd_vel',
                                   track_only_velocity=True,
                                   joint_name=self.drive_joint_name)


class PR2VelocityMujocoInterface(RobotInterfaceConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(self,
                 map_name: str = 'map',
                 localization_joint_name: str = 'localization',
                 odom_link_name: str = 'odom_combined',
                 drive_joint_name: str = 'brumbrum'):
        self.map_name = map_name
        self.localization_joint_name = localization_joint_name
        self.odom_link_name = odom_link_name
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.discover_interfaces_from_controller_manager()
        self.sync_odometry_topic('/odom', self.drive_joint_name)
        self.add_base_cmd_velocity(cmd_vel_topic='/cmd_vel')


@dataclass
class PR2QPControllerConfig(QPControllerConfig):
    def setup(self):
        head_pan_joint: RevoluteConnection = god_map.world.get_connection_by_name('head_pan_joint')
        head_tilt_joint: RevoluteConnection = god_map.world.get_connection_by_name('head_tilt_joint')
        r_shoulder_pan_joint: RevoluteConnection = god_map.world.get_connection_by_name('r_shoulder_pan_joint')
        l_shoulder_pan_joint: RevoluteConnection = god_map.world.get_connection_by_name('l_shoulder_pan_joint')
        r_shoulder_lift_joint: RevoluteConnection = god_map.world.get_connection_by_name('r_shoulder_lift_joint')
        l_shoulder_lift_joint: RevoluteConnection = god_map.world.get_connection_by_name('l_shoulder_lift_joint')

        self.dof_lower_limits_overwrite[head_pan_joint.dof.name].velocity = -1.0
        self.dof_upper_limits_overwrite[head_pan_joint.dof.name].velocity = 1.0

        self.dof_lower_limits_overwrite[head_tilt_joint.dof.name].velocity = -3.5
        self.dof_upper_limits_overwrite[head_tilt_joint.dof.name].velocity = 3.5

        self.dof_lower_limits_overwrite[r_shoulder_pan_joint.dof.name].velocity = -0.15
        self.dof_upper_limits_overwrite[r_shoulder_pan_joint.dof.name].velocity = 0.15
        self.dof_lower_limits_overwrite[l_shoulder_pan_joint.dof.name].velocity = -0.15
        self.dof_upper_limits_overwrite[l_shoulder_pan_joint.dof.name].velocity = 0.15

        self.dof_lower_limits_overwrite[r_shoulder_lift_joint.dof.name].velocity = -0.2
        self.dof_upper_limits_overwrite[r_shoulder_lift_joint.dof.name].velocity = 0.2
        self.dof_lower_limits_overwrite[l_shoulder_lift_joint.dof.name].velocity = -0.2
        self.dof_upper_limits_overwrite[l_shoulder_lift_joint.dof.name].velocity = 0.2
        super().setup()
