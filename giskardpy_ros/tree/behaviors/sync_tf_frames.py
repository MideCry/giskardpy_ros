from typing import Tuple, Dict, Optional

from py_trees.common import Status

from giskardpy.utils.decorators import record_time
from giskardpy_ros.ros2 import msg_converter
from giskardpy_ros.ros2.tfwrapper import lookup_pose
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import (
    catch_and_raise_to_blackboard,
    GiskardBlackboard,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connections import Connection6DoF


class SyncTfFrames(GiskardBehavior):
    joint_map: Dict[PrefixedName, Tuple[str, str]]

    def __init__(
        self, name, joint_map: Optional[Dict[PrefixedName, Tuple[str, str]]] = None
    ):
        super().__init__(name)
        if joint_map is None:
            self.joint_map = {}
        else:
            self.joint_map = joint_map

    def sync_6dof_joint_with_tf_frame(
        self, joint_name: PrefixedName, tf_parent_frame: str, tf_child_frame: str
    ):
        if joint_name in self.joint_map:
            raise AttributeError(
                f"Joint '{joint_name}' is already being tracking with a tf frame: "
                f"'{self.joint_map[joint_name][0]}'<-'{self.joint_map[joint_name][1]}'"
            )
        joint = GiskardBlackboard().executor.world.joints[joint_name]
        if not isinstance(joint, Connection6DoF):
            raise AttributeError(
                f"Can only sync Connection6DoF with tf but '{joint_name}' is of type '{type(joint)}'."
            )
        self.joint_map[joint_name] = (tf_parent_frame, tf_child_frame)

    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        for joint_name, (tf_parent_frame, tf_child_frame) in self.joint_map.items():
            joint: Connection6DoF = GiskardBlackboard().executor.world.joints[
                joint_name
            ]
            parent_T_child = lookup_pose(tf_parent_frame, tf_child_frame)
            pose = msg_converter.ros_msg_to_giskard_obj(
                parent_T_child, GiskardBlackboard().executor.world
            )
            joint.origin = pose

        return Status.SUCCESS
