from typing import Optional

from py_trees.common import Status
from sensor_msgs.msg import JointState

from giskardpy_ros.ros2 import rospy
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard


class PublishJointState(GiskardBehavior):

    def __init__(
        self,
        name: Optional[str] = None,
        topic_name: Optional[str] = None,
        include_prefix: bool = False,
        only_prismatic_and_revolute: bool = True,
    ):
        if name is None:
            name = self.__class__.__name__
        if topic_name is None:
            topic_name = "/joint_states"
        super().__init__(name)
        self.include_prefix = include_prefix
        self.cmd_topic = topic_name
        self.cmd_pub = rospy.node.create_publisher(JointState, self.cmd_topic, 10)
        if only_prismatic_and_revolute:
            self.joint_names = [
                k
                for k in GiskardBlackboard().executor.world.joint_names
                if GiskardBlackboard().executor.world.is_joint_revolute(k)
                or GiskardBlackboard().executor.world.is_joint_prismatic(k)
            ]
        else:
            self.joint_names = list(GiskardBlackboard().executor.world.state.keys())

    def update(self):
        msg = JointState()
        for joint_name in self.joint_names:
            if self.include_prefix:
                msg.name.append(joint_name.long_name)
            else:
                msg.name.append(joint_name.short_name)
            msg.position.append(
                GiskardBlackboard().executor.world.state[joint_name].position
            )
            msg.velocity.append(
                GiskardBlackboard().executor.world.state[joint_name].velocity
            )
        msg.header.stamp = rospy.node.get_clock().now().to_msg()
        self.cmd_pub.publish(msg)
        return Status.SUCCESS
