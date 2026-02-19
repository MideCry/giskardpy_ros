from dataclasses import dataclass, field

from giskardpy_ros.configs.robot_interface_config import StandAloneRobotInterfaceConfig
from giskardpy.configs.world_config import WorldWithDiffDriveRobot
from giskardpy_ros.configs.robot_interface_config import RobotInterfaceConfig
from semantic_digital_twin.robots.stretch import Stretch


class StretchCollisionAvoidanceConfig(CollisionAvoidanceConfig):
    def __init__(self, drive_joint_name: str = 'brumbrum'):
        super().__init__()
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.load_self_collision_matrix('package://giskardpy_ros/self_collision_matrices/iai/stretch.srdf')
        self.overwrite_external_collision_avoidance(self.drive_joint_name,
                                                    number_of_repeller=2,
                                                    soft_threshold=0.2,
                                                    hard_threshold=0.1)


class StretchStandaloneInterface(StandAloneRobotInterfaceConfig):

    def __init__(self, drive_joint_name: str = 'brumbrum'):
        super().__init__([
            drive_joint_name,
            'joint_gripper_finger_left',
            'joint_gripper_finger_right',
            'joint_right_wheel',
            'joint_left_wheel',
            'joint_lift',
            'joint_arm_l3',
            'joint_arm_l2',
            'joint_arm_l1',
            'joint_arm_l0',
            'joint_wrist_yaw',
            'joint_head_pan',
            'joint_head_tilt',
        ])

class StretchVelocityInterface(RobotInterfaceConfig):

    def setup(self):
        # self.sync_6dof_joint_with_tf_frame(
        #     joint=self.world.get_connections_by_type(Connection6DoF)[0],
        #     tf_parent_frame="map",
        #     tf_child_frame="odom",
        # )
        #
        # omni_drive = self.world.get_connections_by_type(OmniDrive)[0]
        # self.sync_odometry_topic(
        #     "/laser_odom",
        #     omni_drive,
        # )
        #
        # self.add_base_cmd_velocity(
        #     cmd_vel_topic="/omni_base_controller/cmd_vel", joint=omni_drive
        # )

        self.sync_joint_state_topic("/joint_states")
        joints = [
            "joint_arm_l0",
            "joint_lift",
            "joint_wrist_yaw",
            "joint_wrist_pitch",
            "joint_wrist_roll",
            "joint_head_pan",
            "joint_head_pan"
            "joint_right_wheel",
            "joint_left_wheel",
            "joint_gripper_finger_left"
        ]
        self.add_joint_velocity_group_controller(
            cmd_topic="/joint_velocity_command", connections=joints
        )

@dataclass
class WorldWithStretchConfig(WorldWithOmniDriveRobot):
    urdf_view: AbstractRobot = field(kw_only=True, default=Stretch, init=False)

    def setup_collision_config(self):
        pass