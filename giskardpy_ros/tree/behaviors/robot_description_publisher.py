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
        description = GiskardBlackboard().giskard.world_config.urdf
        self.msg = String()
        self.msg.data = description

    @record_time
    @profile
    def update(self):
        self.pub.publish(self.msg)
        return Status.SUCCESS
