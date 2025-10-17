from typing import Dict

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped, Quaternion

import giskardpy_ros.ros2.tfwrapper as tf
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.utils.math import quaternion_from_axis_angle
from giskardpy_ros.ros2 import rospy, ros2_interface
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard
from giskardpy_ros.utils.utils import load_xacro
from giskardpy_ros.utils.utils_for_tests import GiskardTester
from semantic_world.world_description.connections import ActiveConnection1DOF


@pytest.fixture(scope="function")
def init_rospy():

    rospy.init_node("giskard")
    get_middleware().loginfo("init ros")
    tf.init()
    get_middleware().loginfo("done tf init")

    try:
        yield None
    finally:
        print("kill ros")
        # Cleanly reset TF and shutdown ROS2 node/executor
        try:
            tf.shutdown()
        except Exception:
            pass
        rospy.shutdown()


@pytest.fixture()
def giskard_factory(init_rospy, robot: GiskardTester):
    def _create_giskard(seed_joint_state: Dict[str, float]) -> GiskardTester:
        seed_joint_state_reached = robot.api.monitors.add_set_seed_configuration(
            seed_joint_state, name="initial joint state"
        )
        if robot.has_odometry_joint():
            zero = PoseStamped()
            zero.header.frame_id = "map"
            zero.pose.orientation.w = 1.0
            base_pose_reached = robot.api.monitors.add_set_seed_odometry(
                zero, name="initial pose"
            )
            done = f"{seed_joint_state_reached} and {base_pose_reached}"
        else:
            done = seed_joint_state_reached
        robot.api.motion_goals.allow_all_collisions()
        robot.api.monitors.add_end_motion(start_condition=done)
        robot.execute()
        return robot

    return _create_giskard


@pytest.fixture()
def giskard(giskard_factory, default_joint_state):
    return giskard_factory(default_joint_state)


@pytest.fixture()
def giskard_better_pose(giskard_factory, better_pose):
    return giskard_factory(better_pose)


@pytest.fixture()
def kitchen_setup(giskard_better_pose: GiskardTester) -> GiskardTester:
    giskard_better_pose.default_env_name = "iai_kitchen"
    kitchen_pose = PoseStamped()
    kitchen_pose.header.frame_id = str(giskard_better_pose.default_root)
    kitchen_pose.pose.orientation.w = 1.0
    kitchen_urdf = load_xacro(
        "package://iai_kitchen/urdf_obj/iai_kitchen_python.urdf.xacro"
    )
    giskard_better_pose.add_urdf_to_world(
        name=giskard_better_pose.default_env_name, urdf=kitchen_urdf, pose=kitchen_pose
    )
    return giskard_better_pose


@pytest.fixture()
def dlr_kitchen_setup(better_pose: GiskardTester) -> GiskardTester:
    better_pose.default_env_name = "dlr_kitchen"
    if GiskardBlackboard().tree_config.is_standalone():
        kitchen_pose = PoseStamped()
        kitchen_pose.header.frame_id = str(better_pose.default_root)
        kitchen_pose.pose.position.x = -2.0
        kitchen_pose.pose.position.y = 2.0
        kitchen_pose.pose.orientation = Quaternion(
            *quaternion_from_axis_angle([0, 0, 1], -np.pi / 2)
        )
        kitchen_urdf = load_xacro(
            "package://iai_kitchen/urdf_obj/iai_kitchen_python.urdf.xacro"
        )
        better_pose.add_urdf_to_world(
            name=better_pose.default_env_name, urdf=kitchen_urdf, pose=kitchen_pose
        )
    else:
        kitchen_pose = tf.lookup_pose("map", "iai_kitchen/world")
        better_pose.add_urdf_to_world(
            name=better_pose.default_env_name,
            urdf=ros2_interface.get_robot_description("kitchen_description"),
            pose=kitchen_pose,
            js_topic="/kitchen/joint_states",
            set_js_topic="/kitchen/cram_joint_states",
        )
    js = {}
    for joint_name in god_map.world.groups[
        better_pose.default_env_name
    ].movable_joint_names:
        joint = god_map.world.joints[joint_name]
        if isinstance(joint, ActiveConnection1DOF):
            if GiskardBlackboard().tree_config.is_standalone():
                js[str(joint.dof.name)] = 0.0
            else:
                js[str(joint.dof.name.name)] = 0.0
    better_pose.set_env_state(js)
    return better_pose


@pytest.fixture()
def apartment_setup(better_pose: GiskardTester) -> GiskardTester:
    better_pose.default_env_name = "iai_apartment"
    if GiskardBlackboard().tree_config.is_standalone():
        kitchen_pose = PoseStamped()
        kitchen_pose.header.frame_id = str(better_pose.default_root)
        kitchen_pose.pose.orientation.w = 1.0
        apartment_urdf = load_xacro("package://iai_apartment/urdf/apartment.urdf")
        better_pose.add_urdf_to_world(
            name=better_pose.default_env_name, urdf=apartment_urdf, pose=kitchen_pose
        )
    else:
        better_pose.add_urdf_to_world(
            name=better_pose.default_env_name,
            urdf=ros2_interface.get_robot_description("apartment_description"),
            pose=tf.lookup_pose("map", "iai_apartment/apartment_root"),
            js_topic="/apartment_joint_states",
            set_js_topic="/iai_kitchen/cram_joint_states",
        )
    base_pose = PoseStamped()
    base_pose.header.frame_id = "side_B"
    base_pose.pose.position.x = 1.5
    base_pose.pose.position.y = 2.4
    base_pose.pose.orientation.w = 1.0
    base_pose = better_pose.transform_msg(god_map.world.root.name, base_pose)
    better_pose.teleport_base(base_pose)
    return better_pose
