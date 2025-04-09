from enum import Enum

from std_msgs.msg import String
from line_profiler import profile
from py_trees.common import Status

from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard
from giskardpy_ros.ros2 import rospy
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from rclpy.qos import QoSProfile, DurabilityPolicy



class RobotDescriptionPublisher(GiskardBehavior):
    """
    Published the robot description.
    """

    @profile
    def __init__(self, name: str, topic: str = 'robot_description'):
        super().__init__(name)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = rospy.node.create_publisher(String, topic, qos)

    @record_time
    @profile
    def update(self):
        description = GiskardBlackboard().giskard.world_config.robot_description
        msg = String()
        msg.data = description
        self.pub.publish(msg)
        return Status.SUCCESS
