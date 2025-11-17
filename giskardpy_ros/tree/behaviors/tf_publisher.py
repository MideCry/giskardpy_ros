from enum import Enum

from geometry_msgs.msg import TransformStamped
from py_trees.common import Status
from tf2_msgs.msg import TFMessage

import giskardpy_ros.ros2.msg_converter as msg_converter
from giskardpy.utils.decorators import record_time
from giskardpy_ros.ros2 import rospy
from giskardpy_ros.ros2.tfwrapper import normalize_quaternion_msg
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard


class TfPublishingModes(Enum):
    nothing = 0
    all = 1
    attached_objects = 2

    world_objects = 4
    attached_and_world_objects = 6


class TFPublisher(GiskardBehavior):
    """
    Published tf for attached and environment objects.
    """

    def __init__(
        self,
        name: str,
        mode: TfPublishingModes,
        tf_topic: str = "tf",
        include_prefix: bool = True,
    ):
        super().__init__(name)
        self.original_links = set(
            body.name for body in GiskardBlackboard().executor.world.bodies
        )
        self.tf_pub = rospy.node.create_publisher(TFMessage, tf_topic, 10)
        self.mode = mode
        self.robots = GiskardBlackboard().giskard.robots
        self.include_prefix = include_prefix

    def make_transform(self, parent_frame, child_frame, pose):
        tf = TransformStamped()
        tf.header.frame_id = parent_frame
        tf.header.stamp = rospy.node.get_clock().now().to_msg()
        tf.child_frame_id = child_frame
        tf.transform.translation.x = pose.position.x
        tf.transform.translation.y = pose.position.y
        tf.transform.translation.z = pose.position.z
        tf.transform.rotation = normalize_quaternion_msg(pose.orientation)
        return tf

    @record_time
    def update(self):
        try:
            if self.mode == TfPublishingModes.all:
                self.tf_pub.publish(
                    msg_converter.world_to_tf_message(
                        GiskardBlackboard().executor.world, self.include_prefix
                    )
                )
            else:
                tf_msg = TFMessage()
                if self.mode in [
                    TfPublishingModes.attached_objects,
                    TfPublishingModes.attached_and_world_objects,
                ]:
                    for robot in self.robots:
                        robot_links = set(robot.bodies)
                    attached_links = robot_links - self.original_links
                    if attached_links:
                        get_fk = GiskardBlackboard().executor.world.compute_fk
                        for body in attached_links:
                            link_name = body.name
                            parent_link_name = body.parent_body
                            fk = get_fk(parent_link_name, link_name)
                            if self.include_prefix:
                                tf = self.make_transform(
                                    fk.header.frame_id, str(link_name), fk.pose
                                )
                            else:
                                tf = self.make_transform(
                                    fk.header.frame_id, str(link_name.name), fk.pose
                                )
                            tf_msg.transforms.append(tf)
            if self.mode in [
                TfPublishingModes.world_objects,
                TfPublishingModes.attached_and_world_objects,
            ]:
                for (
                    group_name,
                    group,
                ) in GiskardBlackboard().executor.world.groups.items():
                    if group_name in self.robots:
                        # robot frames will exist for sure
                        continue
                    if len(group.joints) > 0:
                        continue
                    get_fk = GiskardBlackboard().executor.world.compute_fk
                    fk = get_fk(
                        GiskardBlackboard().executor.world.root.name,
                        group.root_link_name,
                    )
                    tf = self.make_transform(
                        fk.header.frame_id, str(group.root_link_name), fk.pose
                    )
                    tf_msg.transforms.append(tf)
                self.tf_pub.publish(tf_msg)

        except KeyError as e:
            pass
        except UnboundLocalError as e:
            pass
        except ValueError as e:
            pass
        return Status.SUCCESS
