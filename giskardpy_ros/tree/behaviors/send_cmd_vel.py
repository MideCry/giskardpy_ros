from typing import Optional

import numpy as np
from geometry_msgs.msg import Twist
from py_trees.common import Status

from giskardpy.middleware import get_middleware
from giskardpy_ros.ros2 import rospy
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import (
    catch_and_raise_to_blackboard,
    GiskardBlackboard,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName


# can be used during closed-loop control, instead of for tracking a trajectory
class SendCmdVelTwist(GiskardBehavior):
    supported_state_types = [Twist]

    def __init__(self, topic_name: str, joint_name: Optional[PrefixedName] = None):
        super().__init__()
        self.threshold = np.array([0.0, 0.0, 0.0])
        self.cmd_topic = topic_name
        self.vel_pub = rospy.node.create_publisher(Twist, self.cmd_topic, 10)

        self.joint = GiskardBlackboard().executor.world.get_drive_joint(
            joint_name=joint_name
        )
        GiskardBlackboard().executor.world.register_controlled_joints([self.joint.name])
        get_middleware().loginfo(f"Created publisher for {self.cmd_topic}.")

    def solver_cmd_to_twist(self, cmd) -> Twist:
        twist = Twist()
        try:
            twist.linear.x = cmd.free_variable_data[self.joint.x_velocity.name][0]
            if abs(twist.linear.x) < self.threshold[0]:
                twist.linear.x = 0.0
        except:
            twist.linear.x = 0.0
        try:
            twist.linear.y = cmd.free_variable_data[self.joint.y_velocity.name][0]
            if abs(twist.linear.y) < self.threshold[1]:
                twist.linear.y = 0.0
        except:
            twist.linear.y = 0.0
        try:
            twist.angular.z = cmd.free_variable_data[self.joint.yaw.name][0]
            if abs(twist.angular.z) < self.threshold[2]:
                twist.angular.z = 0.0
        except:
            twist.angular.z = 0.0
        return twist

    @catch_and_raise_to_blackboard
    def update(self):
        cmd = GiskardBlackboard().executor.qp_solver_solution
        twist = self.solver_cmd_to_twist(cmd)
        self.vel_pub.publish(twist)
        return Status.RUNNING

    def terminate(self, new_status):
        self.vel_pub.publish(Twist())
        super().terminate(new_status)
