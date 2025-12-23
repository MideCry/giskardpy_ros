from typing import Optional, Tuple

import numpy as np
import giskardpy_ros.ros2.tfwrapper as tf
from giskardpy.data_types.data_types import Derivatives, PrefixName
from giskardpy.data_types.exceptions import UnknownLinkException
from giskardpy.model.joints import RevoluteJoint
from giskardpy.model.world_config import WorldConfig
from giskardpy_ros.configs.giskard import RobotInterfaceConfig
from giskardpy_ros.ros2 import ros2_interface
from giskardpy_ros.ros2.ros2_interface import get_parameters, search_for_unique_publisher_of_type


class TracyVelocityInterface(RobotInterfaceConfig):
    def __init__(self, controller_manager_name: str = 'controller_manager'):
        self.controller_manager_name = controller_manager_name

    def setup(self):
        self.sync_joint_state_topic('/left_arm/joint_states')
        self.sync_joint_state_topic('/right_arm/joint_states')
        self.sync_joint_state_topic('/right_gripper/joint_states')
        self.sync_joint_state_topic('/left_gripper/joint_states')
        joints_left = ['left_shoulder_pan_joint',
                       'left_shoulder_lift_joint',
                       'left_elbow_joint',
                       'left_wrist_1_joint',
                       'left_wrist_2_joint',
                       'left_wrist_3_joint']
        self.add_joint_velocity_group_controller(cmd_topic='/left_arm/forward_velocity_controller/commands', joints=joints_left)
        joints_right = ['right_shoulder_pan_joint',
                       'right_shoulder_lift_joint',
                       'right_elbow_joint',
                       'right_wrist_1_joint',
                       'right_wrist_2_joint',
                       'right_wrist_3_joint']
        self.add_joint_velocity_group_controller(cmd_topic='/right_arm/forward_velocity_controller/commands', joints=joints_right)



class TracyWorldConfig(WorldConfig):
    robot_name: str = ''
    robot_description: Optional[str]
    controller_manager_name: str

    def __init__(self,
                 robot_description: Optional[str] = None,
                 controller_manager_name: str = 'controller_manager'):
        super().__init__()
        self.robot_description = robot_description
        self.controller_manager_name = controller_manager_name

    def get_tf_root_that_is_not_in_world(self) -> str:
        tf_roots = set(tf.get_tf_roots())
        if len(tf_roots) == 1:
            return tf_roots.pop()
        frames_not_in_world = tf_roots.difference(self.world.link_names_as_set)
        if len(frames_not_in_world) > 0:
            return frames_not_in_world.pop()
        return self.world.groups[list(self.world.group_names)[0]].root_link_name.short_name

    def setup(self):
        self.set_default_limits({Derivatives.velocity: 0.2,
                                 Derivatives.acceleration: np.inf,
                                 Derivatives.jerk: None})
        global_tf_frame = self.get_tf_root_that_is_not_in_world()
        self.map_name = PrefixName(global_tf_frame)
        self.urdf = self.robot_description or ros2_interface.get_robot_description()
        self.add_robot_urdf(self.urdf, self.robot_name)

        # limit last joints based on cable limits: doesn't work
        # joint_name = self.world.search_for_joint_name('left_wrist_3_joint')
        # joint: RevoluteJoint = self.world.joints[joint_name]
        # joint.free_variable.set_upper_limit(Derivatives.position, -1.54)
        # joint.free_variable.set_lower_limit(Derivatives.position, -2.28)
        #
        # joint_name = self.world.search_for_joint_name('right_wrist_3_joint')
        # joint: RevoluteJoint = self.world.joints[joint_name]
        # joint.free_variable.set_upper_limit(Derivatives.position, -2.5)
        # joint.free_variable.set_lower_limit(Derivatives.position, -4.21)

        root_link_name = self.get_root_link_of_group(self.robot_name)
        # gather frames between tf root and robot root
        chain = tf.get_frame_chain(global_tf_frame, root_link_name.short_name)
        # add all missing frames to world
        for i, tf_frame in enumerate(chain):
            try:
                world_link_name = self.world.search_for_link_name(tf_frame)
            except UnknownLinkException as e:
                world_link_name = PrefixName(tf_frame)
                self.add_empty_link(world_link_name)
            chain[i] = world_link_name
        for link1, link2 in zip(chain, chain[1:]):
            self.add_fixed_joint(parent_link=link1, child_link=link2)

