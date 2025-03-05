
from copy import deepcopy
from typing import Optional

import rospy
from line_profiler import profile
from sensor_msgs.msg import JointState
from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior


class PublishJointState(GiskardBehavior):
    @profile
    def __init__(self, name: Optional[str] = None, topic_name: Optional[str] = None, include_prefix: bool = False,
                 only_prismatic_and_revolute: bool = True, group_name: Optional[str] = None):
        if name is None:
            name = self.__class__.__name__
        if topic_name is None:
            topic_name = '/joint_states'
        super().__init__(name)
        self.include_prefix = include_prefix
        self.cmd_topic = topic_name
        self.cmd_pub = rospy.Publisher(self.cmd_topic, JointState, queue_size=10)
        if group_name is None:
            group = god_map.world
        else:
            group = god_map.world.groups[group_name]
        if only_prismatic_and_revolute:
            self.joint_names = [k for k in group.joint_names if group.is_joint_revolute(k)
                                or group.is_joint_prismatic(k)]
        else:
            self.joint_names = list(group.state.keys())

    def update(self):
        msg = JointState()
        for joint_name in self.joint_names:
            if self.include_prefix:
                msg.name.append(joint_name.long_name)
            else:
                msg.name.append(joint_name.short_name)
            msg.position.append(god_map.world.state[joint_name].position)
            msg.velocity.append(god_map.world.state[joint_name].velocity)
        msg.header.stamp = rospy.get_rostime()
        self.cmd_pub.publish(msg)
        return Status.SUCCESS
