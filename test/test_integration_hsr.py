from copy import deepcopy

import numpy as np
import pytest
from geometry_msgs.msg import (
    PoseStamped,
    Point,
    Quaternion,
    PointStamped,
    Vector3Stamped,
)
from numpy import pi

from giskardpy.data_types.exceptions import EmptyProblemException
from giskardpy.model.collision_world_syncer import CollisionCheckerLib
from giskardpy.motion_statechart.goals.test import GraspSequence, Cutting
from giskardpy.motion_statechart.tasks.pointing import PointingCone
from giskardpy.motion_statechart.test_nodes.test_nodes import ConstTrueNode
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.utils.math import (
    quaternion_from_axis_angle,
    quaternion_from_rotation_matrix,
)
from giskardpy_ros.configs.behavior_tree_config import StandAloneBTConfig
from giskardpy_ros.configs.giskard import Giskard
from giskardpy_ros.configs.iai_robots.hsr import (
    WorldWithHSRConfig,
    HSRStandaloneInterface,
)
from giskardpy_ros.utils.utils import load_xacro
from giskardpy_ros.utils.utils_for_tests import compare_poses, GiskardTester
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF


@pytest.fixture()
def default_joint_state():
    return {
        "arm_flex_joint": -0.03,
        "arm_lift_joint": 0.01,
        "arm_roll_joint": 0.0,
        "head_pan_joint": 0.0,
        "head_tilt_joint": 0.0,
        "wrist_flex_joint": 0.0,
        "wrist_roll_joint": 0.0,
    }


class HSRTester(GiskardTester):

    def __init__(self, giskard=None):
        self.tip = "hand_gripper_tool_frame"
        if giskard is None:
            robot_desc = load_xacro(
                "package://hsr_description/robots/hsrb4s.urdf.xacro"
            )
            giskard = Giskard(
                world_config=WorldWithHSRConfig(urdf=robot_desc),
                robot_interface_config=HSRStandaloneInterface(),
                collision_checker_id=CollisionCheckerLib.bpb,
                behavior_tree_config=StandAloneBTConfig(
                    debug_mode=True,
                    publish_tf=True,
                    publish_js=False,
                    add_debug_marker_publisher=True,
                ),
                qp_controller_config=QPControllerConfig(mpc_dt=0.05, control_dt=None),
            )
        super().__init__(giskard)

    def open_gripper(self):
        self.command_gripper(1.23)

    def close_gripper(self):
        self.command_gripper(0)

    def command_gripper(self, width):
        js = {"hand_motor_joint": width}
        self.api.monitors.add_set_seed_configuration(
            seed_configuration=js, name="move gripper"
        )
        self.execute()


@pytest.fixture()
def robot():
    c = HSRTester()
    try:
        yield c
    finally:
        print("tear down")
        c.print_stats()


# @pytest.fixture(scope="module")
# def giskard(request, ros):
#     # launch_launchfile('package://hsr_description/launch/upload_hsrb.launch')
#     c = HSRTester()
#     # c = HSRTestWrapperMujoco()
#     request.addfinalizer(c.print_stats)
#     return c
#
#
# @pytest.fixture()
# def box_setup(default_pose_giskard: HSRTester) -> HSRTester:
#     p = PoseStamped()
#     p.header.frame_id = "map"
#     p.pose.position.x = 1.2
#     p.pose.position.y = 0.0
#     p.pose.position.z = 0.1
#     p.pose.orientation.w = 1.0
#     default_pose_giskard.add_box_to_world(name="box", size=(1, 1, 1), pose=p)
#     return default_pose_giskard


class TestJointGoals:

    def test_mimic_joints(self, default_pose_giskard: HSRTester):
        arm_lift_joint = default_pose_giskard.api.world.get_connection_by_name(
            "arm_lift_joint"
        )
        default_pose_giskard.open_gripper()
        hand_T_finger_current = default_pose_giskard.compute_fk_pose(
            "hand_palm_link", "hand_l_distal_link"
        )
        hand_T_finger_expected = PoseStamped()
        hand_T_finger_expected.header.frame_id = "hand_palm_link"
        hand_T_finger_expected.pose.position.x = -0.01675
        hand_T_finger_expected.pose.position.y = -0.0907
        hand_T_finger_expected.pose.position.z = 0.0052
        hand_T_finger_expected.pose.orientation.x = -0.0434
        hand_T_finger_expected.pose.orientation.y = 0.0
        hand_T_finger_expected.pose.orientation.z = 0.0
        hand_T_finger_expected.pose.orientation.w = 0.999
        compare_poses(hand_T_finger_current.pose, hand_T_finger_expected.pose)

        js = {"torso_lift_joint": 0.1}
        default_pose_giskard.api.motion_goals.add_joint_position(js)
        default_pose_giskard.api.motion_goals.allow_all_collisions()
        default_pose_giskard.execute()
        np.testing.assert_almost_equal(
            default_pose_giskard.api.world.state[arm_lift_joint.dof.name].position,
            0.2,
            decimal=2,
        )
        base_T_torso = PoseStamped()
        base_T_torso.header.frame_id = "base_footprint"
        base_T_torso.pose.position.x = 0.0
        base_T_torso.pose.position.y = 0.0
        base_T_torso.pose.position.z = 0.8518
        base_T_torso.pose.orientation.x = 0.0
        base_T_torso.pose.orientation.y = 0.0
        base_T_torso.pose.orientation.z = 0.0
        base_T_torso.pose.orientation.w = 1.0
        base_T_torso2 = default_pose_giskard.compute_fk_pose(
            "base_footprint", "torso_lift_link"
        )
        compare_poses(base_T_torso2.pose, base_T_torso.pose)

    def test_mimic_joints2(self, default_pose_giskard: HSRTester):
        arm_lift_joint = default_pose_giskard.api.world.get_connection_by_name(
            "arm_lift_joint"
        )
        default_pose_giskard.open_gripper()

        tip = "hand_gripper_tool_frame"
        p = PoseStamped()
        p.header.frame_id = tip
        p.pose.position.z = 0.2
        p.pose.orientation.w = 1.0
        default_pose_giskard.api.motion_goals.add_cartesian_pose(
            goal_pose=p, tip_link=tip, root_link="base_footprint"
        )
        default_pose_giskard.api.motion_goals.allow_all_collisions()
        default_pose_giskard.execute()
        np.testing.assert_almost_equal(
            default_pose_giskard.api.world.state[arm_lift_joint.dof.name].position,
            0.2,
            decimal=2,
        )
        base_T_torso = PoseStamped()
        base_T_torso.header.frame_id = "base_footprint"
        base_T_torso.pose.position.x = 0.0
        base_T_torso.pose.position.y = 0.0
        base_T_torso.pose.position.z = 0.8518
        base_T_torso.pose.orientation.x = 0.0
        base_T_torso.pose.orientation.y = 0.0
        base_T_torso.pose.orientation.z = 0.0
        base_T_torso.pose.orientation.w = 1.0
        base_T_torso2 = default_pose_giskard.compute_fk_pose(
            "base_footprint", "torso_lift_link"
        )
        compare_poses(base_T_torso2.pose, base_T_torso.pose)

    def test_mimic_joints3(self, default_pose_giskard: HSRTester):
        arm_lift_joint = default_pose_giskard.api.world.get_connection_by_name(
            "arm_lift_joint"
        )
        default_pose_giskard.open_gripper()
        tip = "head_pan_link"
        p = PoseStamped()
        p.header.frame_id = tip
        p.pose.position.z = 0.15
        p.pose.orientation.w = 1.0
        default_pose_giskard.api.motion_goals.add_cartesian_pose(
            goal_pose=p, tip_link=tip, root_link="base_footprint"
        )
        default_pose_giskard.execute()
        np.testing.assert_almost_equal(
            default_pose_giskard.api.world.state[arm_lift_joint.dof.name].position,
            0.3,
            decimal=2,
        )
        base_T_torso = PoseStamped()
        base_T_torso.header.frame_id = "base_footprint"
        base_T_torso.pose.position.x = 0.0
        base_T_torso.pose.position.y = 0.0
        base_T_torso.pose.position.z = 0.902
        base_T_torso.pose.orientation.x = 0.0
        base_T_torso.pose.orientation.y = 0.0
        base_T_torso.pose.orientation.z = 0.0
        base_T_torso.pose.orientation.w = 1.0
        base_T_torso2 = default_pose_giskard.compute_fk_pose(
            "base_footprint", "torso_lift_link"
        )
        compare_poses(base_T_torso2.pose, base_T_torso.pose)

    def test_mimic_joints4(self, default_pose_giskard: HSRTester):
        arm_lift_joints: ActiveConnection1DOF = (
            default_pose_giskard.apdefault_pose_giskard.apdefault_pose_giskard.apdefault_pose_giskard.api.world.get_connection_by_name(
                "arm_lift_joint"
            )
        )
        assert arm_lift_joints.dof.lower_limits.velocity == -0.15
        assert arm_lift_joints.dof.upper_limits.velocity == 0.15
        torso_lift_joints: ActiveConnection1DOF = (
            default_pose_giskard.api.world.get_connection_by_name("torso_lift_joint")
        )
        assert torso_lift_joints.dof.lower_limits.velocity == -0.075
        assert torso_lift_joints.dof.upper_limits.velocity == 0.075
        joint_goal = {"torso_lift_joint": 0.25}
        default_pose_giskard.api.motion_goals.add_joint_position(joint_goal)
        default_pose_giskard.api.motion_goals.allow_all_collisions()
        default_pose_giskard.execute()
        np.testing.assert_almost_equal(
            default_pose_giskard.api.world.state[arm_lift_joints.dof.name].position,
            0.5,
            decimal=2,
        )


class TestCartGoals:
    def test_move_base(self, default_pose_giskard: HSRTester):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = "map"
        map_T_odom.pose.position.x = 1.0
        map_T_odom.pose.position.y = 1.0
        q = quaternion_from_axis_angle([0, 0, 1], np.pi / 3)
        map_T_odom.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        default_pose_giskard.teleport_base(map_T_odom)

        base_goal = PoseStamped()
        base_goal.header.frame_id = "map"
        base_goal.pose.position.x = 1.0
        q = quaternion_from_axis_angle([0, 0, 1], pi)
        base_goal.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        default_pose_giskard.api.motion_goals.add_cartesian_pose(
            goal_pose=base_goal, tip_link="base_footprint", root_link="map"
        )
        default_pose_giskard.api.motion_goals.allow_all_collisions()
        default_pose_giskard.execute()

    def test_move_base_1m_forward(self, default_pose_giskard: HSRTester):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = "map"
        map_T_odom.pose.position.x = 1.0
        map_T_odom.pose.orientation.w = 1.0
        default_pose_giskard.api.motion_goals.allow_all_collisions()
        default_pose_giskard.move_base(map_T_odom)

    def test_move_base_1m_left(self, default_pose_giskard: HSRTester):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = "map"
        map_T_odom.pose.position.y = 1.0
        map_T_odom.pose.orientation.w = 1.0
        default_pose_giskard.api.motion_goals.allow_all_collisions()
        default_pose_giskard.move_base(map_T_odom)

    def test_move_base_1m_diagonal(self, default_pose_giskard: HSRTester):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = "map"
        map_T_odom.pose.position.x = 1.0
        map_T_odom.pose.position.y = 1.0
        map_T_odom.pose.orientation.w = 1.0
        default_pose_giskard.api.motion_goals.allow_all_collisions()
        default_pose_giskard.move_base(map_T_odom)

    def test_move_base_rotate(self, default_pose_giskard: HSRTester):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = "map"
        q = quaternion_from_axis_angle([0, 0, 1], np.pi / 3)
        map_T_odom.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        default_pose_giskard.api.motion_goals.allow_all_collisions()
        default_pose_giskard.move_base(map_T_odom)

    def test_move_base_forward_rotate(self, default_pose_giskard: HSRTester):
        map_T_odom = PoseStamped()
        map_T_odom.header.frame_id = "map"
        map_T_odom.pose.position.x = 1.0
        q = quaternion_from_axis_angle([0, 0, 1], np.pi / 3)
        map_T_odom.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        default_pose_giskard.api.motion_goals.allow_all_collisions()
        default_pose_giskard.move_base(map_T_odom)

    def test_rotate_gripper(self, default_pose_giskard: HSRTester):
        r_goal = PoseStamped()
        r_goal.header.frame_id = default_pose_giskard.tip
        q = quaternion_from_axis_angle([0, 0, 1], pi)
        r_goal.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        default_pose_giskard.api.motion_goals.add_cartesian_pose(
            goal_pose=r_goal, tip_link=default_pose_giskard.tip, root_link="map"
        )
        default_pose_giskard.api.motion_goals.allow_all_collisions()
        default_pose_giskard.execute()

    def test_wiggle_insert(self, default_pose_giskard: HSRTester):
        goal_state = {
            "arm_flex_joint": -1.5,
            "arm_lift_joint": 0.5,
            "arm_roll_joint": 0.0,
            "head_pan_joint": 0.0,
            "head_tilt_joint": 0.0,
            "wrist_flex_joint": -1.5,
            "wrist_roll_joint": 0.0,
        }

        default_pose_giskard.api.monitors.add_set_seed_configuration(
            seed_configuration=goal_state
        )
        default_pose_giskard.execute()

        hpl = (
            default_pose_giskard.apdefault_pose_giskard.api.world.search_for_link_name(
                link_name="hand_gripper_tool_frame", group_name="hsrb"
            )
        )
        root_link = default_pose_giskard.api.world.search_for_link_name(link_name="map")
        hole_point = PointStamped()
        hole_point.header.frame_id = "map"
        hole_point.point.x = 0.5
        hole_point.point.z = 0.3
        wiggle = "wiggle"
        default_pose_giskard.api.motion_goals.add_wiggle_insert(
            name=wiggle,
            root_link=root_link,
            tip_link=hpl,
            hole_point=hole_point,
            end_condition=wiggle,
        )
        resistence_point = PointStamped()
        resistence_point.header.frame_id = "map"
        resistence_point.point.x = 0.5
        resistence_point.point.z = 0.4
        timer = default_pose_giskard.api.monitors.add_sleep(5)
        default_pose_giskard.api.motion_goals.add_cartesian_position(
            root_link=root_link,
            tip_link=hpl,
            goal_point=resistence_point,
            end_condition=timer,
        )
        default_pose_giskard.api.monitors.add_end_motion(start_condition=wiggle)
        default_pose_giskard.execute(local_min_end=False)


class TestConstraints:

    def test_Pointing(self, giskard: HSRTester):
        kopf = "head_rgbd_sensor_gazebo_frame"

        head_goal_point = PointStamped()
        head_goal_point.header.frame_id = "map"
        head_goal_point.point.x = 1.0
        head_goal_point.point.y = -1.0
        head_goal_point.point.z = 0.0

        pointing_goal = "pointing"

        pointing_axis = Vector3Stamped()
        pointing_axis.header.frame_id = kopf
        pointing_axis.vector.z = 1.0

        giskard.api.motion_goals.add_pointing(
            name=pointing_goal,
            pointing_axis=pointing_axis,
            root_link="map",
            tip_link=kopf,
            goal_point=head_goal_point,
        )

        giskard.api.motion_goals.allow_all_collisions()
        giskard.api.add_default_end_motion_conditions()
        giskard.execute(local_min_end=False)

    def test_PointingCone(self, default_pose_giskard: HSRTester):
        tip_link = "head_center_camera_frame"
        goal_point = PointStamped()
        goal_point.header.frame_id = "map"
        goal_point.point.x = 0.5
        goal_point.point.y = -0.5
        goal_point.point.z = 1.0

        pointing_axis = Vector3Stamped()
        pointing_axis.header.frame_id = tip_link
        pointing_axis.vector.z = 1.0

        default_pose_giskard.api.motion_goals.add_motion_goal(
            class_name=PointingCone.__name__,
            name="pointy_cone",
            tip_link=tip_link,
            root_link="map",
            goal_point=goal_point,
            pointing_axis=pointing_axis,
        )
        default_pose_giskard.api.motion_goals.allow_all_collisions()
        default_pose_giskard.api.add_default_end_motion_conditions()
        default_pose_giskard.execute(local_min_end=False)

    def test_open_fridge(self, kitchen_setup: HSRTester):
        handle_frame_id = "iai_fridge_door_handle"
        handle_name = "iai_fridge_door_handle"
        kitchen_setup.open_gripper()
        base_goal = PoseStamped()
        base_goal.header.frame_id = "map"
        base_goal.pose.position = Point(x=0.3, y=-0.5, z=0.0)
        base_goal.pose.orientation.w = 1.0
        kitchen_setup.move_base(base_goal)

        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.z = 1.0

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.tip
        tip_grasp_axis.vector.x = 1.0

        kitchen_setup.api.motion_goals.add_grasp_bar(
            root_link=kitchen_setup.default_root,
            tip_link=kitchen_setup.tip,
            tip_grasp_axis=tip_grasp_axis,
            bar_center=bar_center,
            bar_axis=bar_axis,
            bar_length=0.4,
        )
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.tip
        x_gripper.vector.z = 1.0

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = handle_frame_id
        x_goal.vector.x = -1.0
        kitchen_setup.api.motion_goals.add_align_planes(
            tip_link=kitchen_setup.tip,
            tip_normal=x_gripper,
            goal_normal=x_goal,
            root_link="map",
        )
        kitchen_setup.api.motion_goals.allow_all_collisions()
        # kitchen_setup.add_json_goal('AvoidJointLimits', percentage=10)
        kitchen_setup.execute()
        current_pose = kitchen_setup.compute_fk_pose(
            root_link="map", tip_link=kitchen_setup.tip
        )

        kitchen_setup.api.motion_goals.add_open_container(
            tip_link=kitchen_setup.tip,
            environment_link=handle_name,
            goal_joint_state=1.5,
        )
        # kitchen_setup.set_json_goal('AvoidJointLimits', percentage=40)
        kitchen_setup.api.motion_goals.allow_all_collisions()
        # kitchen_setup.add_json_goal('AvoidJointLimits')
        kitchen_setup.execute()
        kitchen_setup.set_env_state({"iai_fridge_door_joint": 1.5})

        pose_reached = kitchen_setup.api.monitors.add_cartesian_pose(
            "map", tip_link=kitchen_setup.tip, goal_pose=current_pose
        )
        kitchen_setup.api.monitors.add_end_motion(start_condition=pose_reached)

        kitchen_setup.api.motion_goals.add_open_container(
            tip_link=kitchen_setup.tip,
            environment_link=handle_name,
            goal_joint_state=0.0,
        )
        kitchen_setup.api.motion_goals.allow_all_collisions()
        # kitchen_setup.set_json_goal('AvoidJointLimits', percentage=40)

        kitchen_setup.execute()

        kitchen_setup.set_env_state({"iai_fridge_door_joint": 0.0})

        kitchen_setup.api.motion_goals.add_joint_position(kitchen_setup.better_pose)
        kitchen_setup.api.motion_goals.allow_self_collision()
        kitchen_setup.execute()

    def test_open_fridge_sequence_simple(self, kitchen_setup: HSRTester):
        handle_frame_id = "iai_fridge_door_handle"
        handle_name = "iai_fridge_door_handle"
        camera_link = "head_rgbd_sensor_link"
        kitchen_setup.open_gripper()
        base_goal = PoseStamped()
        base_goal.header.frame_id = "map"
        base_goal.pose.position = Point(x=0.3, y=-0.5, z=0.0)
        base_goal.pose.orientation.w = 1.0
        kitchen_setup.api.motion_goals.allow_all_collisions()
        kitchen_setup.move_base(base_goal)

        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.z = 1.0

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.tip
        tip_grasp_axis.vector.x = 1.0

        # %% phase 1 grasp handle
        bar_grasped = kitchen_setup.api.motion_goals.add_grasp_bar(
            root_link=kitchen_setup.default_root,
            tip_link=kitchen_setup.tip,
            tip_grasp_axis=tip_grasp_axis,
            bar_center=bar_center,
            bar_axis=bar_axis,
            bar_length=0.4,
            name="grasp handle",
        )

        # %% close gripper
        gripper_closed = kitchen_setup.api.motion_goals.add_joint_position(
            name="close gripper", goal_state={"hand_motor_joint": 0.0}
        )
        gripper_opened = kitchen_setup.api.motion_goals.add_joint_position(
            name="open gripper", goal_state={"hand_motor_joint": 1.23}
        )

        # %% phase 2 open door
        door_open = kitchen_setup.api.motion_goals.add_open_container(
            tip_link=kitchen_setup.tip,
            environment_link=handle_name,
            goal_joint_state=1.5,
            name="open door",
        )

        kitchen_setup.api.update_end_condition(
            node_name=bar_grasped, condition=bar_grasped
        )

        kitchen_setup.api.update_start_condition(
            node_name=gripper_closed, condition=bar_grasped
        )
        kitchen_setup.api.update_end_condition(
            node_name=gripper_closed, condition=gripper_closed
        )

        kitchen_setup.api.update_start_condition(
            node_name=door_open, condition=gripper_closed
        )
        kitchen_setup.api.update_start_condition(
            node_name=gripper_opened, condition=f"{door_open}"
        )

        kitchen_setup.api.update_end_condition(
            node_name=door_open, condition=f"{door_open}"
        )

        kitchen_setup.api.motion_goals.allow_all_collisions()
        kitchen_setup.api.monitors.add_end_motion(start_condition=f"{gripper_opened}")
        kitchen_setup.execute(local_min_end=False)

    def test_open_fridge_sequence_semi_simple(self, kitchen_setup: HSRTester):
        handle_frame_id = "iai_fridge_door_handle"
        handle_name = "iai_fridge_door_handle"
        camera_link = "head_rgbd_sensor_link"
        kitchen_setup.open_gripper()
        base_goal = PoseStamped()
        base_goal.header.frame_id = "map"
        base_goal.pose.position = Point(x=0.3, y=-0.5, z=0.0)
        base_goal.pose.orientation.w = 1.0
        kitchen_setup.api.motion_goals.allow_all_collisions()
        kitchen_setup.move_base(base_goal)

        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.z = 1.0

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.tip
        tip_grasp_axis.vector.x = 1.0

        handle_detected = kitchen_setup.api.monitors.add_const_true(
            name="Detect Handle"
        )

        # %% phase 1 grasp handle
        laser_violated = kitchen_setup.api.monitors.add_pulse(
            after_ticks=20, name="laser violated"
        )
        camera_z = Vector3Stamped()
        camera_z.header.frame_id = camera_link
        camera_z.vector.z = 1.0

        bar_grasped = kitchen_setup.api.motion_goals.add_grasp_bar(
            root_link=kitchen_setup.default_root,
            tip_link=kitchen_setup.tip,
            tip_grasp_axis=tip_grasp_axis,
            bar_center=bar_center,
            bar_axis=bar_axis,
            bar_length=0.4,
            name="grasp handle",
        )

        # %% close gripper
        gripper_closed = kitchen_setup.api.motion_goals.add_joint_position(
            name="close gripper", goal_state={"hand_motor_joint": 0.0}
        )
        gripper_opened = kitchen_setup.api.motion_goals.add_joint_position(
            name="open gripper", goal_state={"hand_motor_joint": 1.23}
        )

        # %% phase 2 open door
        slipped = kitchen_setup.api.monitors.add_pulse(name="slipped", after_ticks=20)
        door_open = kitchen_setup.api.motion_goals.add_open_container(
            tip_link=kitchen_setup.tip,
            environment_link=handle_name,
            goal_joint_state=1.5,
            name="open door",
        )
        reset = kitchen_setup.api.monitors.add_monitor(
            class_name=ConstTrueNode.__name__,
            name="The Great Reset",
        )

        kitchen_setup.api.update_start_condition(
            node_name=laser_violated, condition=handle_detected
        )
        kitchen_setup.api.update_start_condition(
            node_name=bar_grasped, condition=handle_detected
        )
        kitchen_setup.api.update_end_condition(
            node_name=gripper_closed, condition=gripper_closed
        )

        kitchen_setup.api.update_end_condition(
            node_name=laser_violated, condition=bar_grasped
        )
        kitchen_setup.api.update_end_condition(
            node_name=bar_grasped, condition=bar_grasped
        )
        kitchen_setup.api.update_pause_condition(
            node_name=bar_grasped, condition=laser_violated
        )

        kitchen_setup.api.update_start_condition(
            node_name=gripper_closed, condition=bar_grasped
        )
        kitchen_setup.api.update_start_condition(
            node_name=door_open, condition=gripper_closed
        )
        kitchen_setup.api.update_start_condition(
            node_name=slipped, condition=gripper_closed
        )
        kitchen_setup.api.update_start_condition(
            node_name=gripper_opened, condition=f"{slipped} or {door_open}"
        )
        kitchen_setup.api.update_end_condition(
            node_name=slipped, condition=f"{slipped} or {door_open}"
        )

        kitchen_setup.api.update_end_condition(
            node_name=door_open, condition=f"{slipped} or {door_open}"
        )
        reset_condition = f"{gripper_opened} and {slipped}"
        kitchen_setup.api.update_start_condition(
            node_name=reset, condition=reset_condition
        )

        kitchen_setup.api.update_reset_condition(node_name=bar_grasped, condition=reset)
        kitchen_setup.api.update_reset_condition(
            node_name=laser_violated, condition=reset
        )
        kitchen_setup.api.update_reset_condition(
            node_name=gripper_closed, condition=reset
        )
        kitchen_setup.api.update_reset_condition(node_name=door_open, condition=reset)
        kitchen_setup.api.update_reset_condition(node_name=slipped, condition=reset)
        kitchen_setup.api.update_reset_condition(
            node_name=gripper_opened, condition=reset
        )
        kitchen_setup.api.update_reset_condition(node_name=reset, condition=reset)

        kitchen_setup.api.motion_goals.allow_all_collisions()
        kitchen_setup.api.monitors.add_end_motion(
            start_condition=f"{door_open} and {gripper_opened} and not {slipped}"
        )
        kitchen_setup.execute(local_min_end=False)

    def test_open_fridge_sequence(self, kitchen_setup: HSRTester):
        handle_frame_id = "iai_fridge_door_handle"
        handle_name = "iai_fridge_door_handle"
        camera_link = "head_rgbd_sensor_link"
        kitchen_setup.open_gripper()
        base_goal = PoseStamped()
        base_goal.header.frame_id = "map"
        base_goal.pose.position = Point(x=0.3, y=-0.5, z=0.0)
        base_goal.pose.orientation.w = 1.0
        kitchen_setup.api.motion_goals.allow_all_collisions()
        kitchen_setup.move_base(base_goal)

        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.z = 1.0

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.tip
        tip_grasp_axis.vector.x = 1.0

        # %% phase 1 grasp handle
        laser_violated = kitchen_setup.api.monitors.add_pulse(
            after_ticks=20, name="laser violated"
        )
        camera_z = Vector3Stamped()
        camera_z.header.frame_id = camera_link
        camera_z.vector.z = 1.0
        pointing_at = kitchen_setup.api.motion_goals.add_pointing(
            goal_point=bar_center,
            tip_link=camera_link,
            name="look at handle",
            root_link="torso_lift_link",
            pointing_axis=camera_z,
        )

        bar_grasped = kitchen_setup.api.motion_goals.add_grasp_bar(
            root_link=kitchen_setup.default_root,
            tip_link=kitchen_setup.tip,
            tip_grasp_axis=tip_grasp_axis,
            bar_center=bar_center,
            bar_axis=bar_axis,
            bar_length=0.4,
            name="grasp handle",
        )
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.tip
        x_gripper.vector.z = 1.0

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = handle_frame_id
        x_goal.vector.x = -1.0
        align_planes = kitchen_setup.api.motion_goals.add_align_planes(
            tip_link=kitchen_setup.tip,
            tip_normal=x_gripper,
            goal_normal=x_goal,
            root_link="map",
            name="align gripper",
        )

        # %% close gripper
        gripper_closed = "close gripper"
        kitchen_setup.api.motion_goals.add_joint_position(
            name=gripper_closed,
            goal_state={"hand_motor_joint": 0.0},
            end_condition=gripper_closed,
        )
        gripper_opened = kitchen_setup.api.motion_goals.add_joint_position(
            name="open gripper", goal_state={"hand_motor_joint": 1.23}
        )

        # %% phase 2 open door
        slipped = kitchen_setup.api.monitors.add_pulse(name="slipped", after_ticks=20)
        door_open = kitchen_setup.api.motion_goals.add_open_container(
            tip_link=kitchen_setup.tip,
            environment_link=handle_name,
            goal_joint_state=1.5,
            name="open door",
        )
        reset = kitchen_setup.api.monitors.add_monitor(
            class_name=ConstTrueNode.__name__, name="The Great Reset"
        )

        kitchen_setup.api.update_start_condition(
            node_name=bar_grasped, condition=pointing_at
        )
        kitchen_setup.api.update_start_condition(
            node_name=align_planes, condition=pointing_at
        )
        kitchen_setup.api.update_start_condition(
            node_name=laser_violated, condition=pointing_at
        )

        kitchen_setup.api.update_end_condition(
            node_name=pointing_at, condition=bar_grasped
        )
        kitchen_setup.api.update_end_condition(
            node_name=align_planes, condition=bar_grasped
        )
        kitchen_setup.api.update_end_condition(
            node_name=laser_violated, condition=bar_grasped
        )
        kitchen_setup.api.update_end_condition(
            node_name=bar_grasped, condition=bar_grasped
        )
        kitchen_setup.api.update_pause_condition(
            node_name=bar_grasped, condition=laser_violated
        )

        kitchen_setup.api.update_start_condition(
            node_name=gripper_closed, condition=bar_grasped
        )

        kitchen_setup.api.update_start_condition(
            node_name=door_open, condition=gripper_closed
        )
        kitchen_setup.api.update_start_condition(
            node_name=slipped, condition=gripper_closed
        )
        kitchen_setup.api.update_start_condition(
            node_name=gripper_opened, condition=f"{slipped} or {door_open}"
        )
        kitchen_setup.api.update_end_condition(
            node_name=slipped, condition=f"{slipped} or {door_open}"
        )

        kitchen_setup.api.update_end_condition(
            node_name=door_open, condition=f"{slipped} or {door_open}"
        )
        reset_condition = f"{gripper_opened} and {slipped}"
        kitchen_setup.api.update_start_condition(
            node_name=reset, condition=reset_condition
        )

        kitchen_setup.api.update_reset_condition(node_name=pointing_at, condition=reset)
        kitchen_setup.api.update_reset_condition(node_name=bar_grasped, condition=reset)
        kitchen_setup.api.update_reset_condition(
            node_name=align_planes, condition=reset
        )
        kitchen_setup.api.update_reset_condition(
            node_name=laser_violated, condition=reset
        )
        kitchen_setup.api.update_reset_condition(
            node_name=gripper_closed, condition=reset
        )
        kitchen_setup.api.update_reset_condition(node_name=door_open, condition=reset)
        kitchen_setup.api.update_reset_condition(node_name=slipped, condition=reset)
        kitchen_setup.api.update_reset_condition(
            node_name=gripper_opened, condition=reset
        )
        kitchen_setup.api.update_reset_condition(node_name=reset, condition=reset)

        kitchen_setup.api.motion_goals.allow_all_collisions()
        kitchen_setup.api.monitors.add_end_motion(
            start_condition=f"{door_open} and {gripper_opened} and not {slipped}"
        )
        kitchen_setup.execute(local_min_end=False)


class TestCollisionAvoidanceGoals:

    def test_self_collision_avoidance_empty(self, default_pose_giskard: HSRTester):
        default_pose_giskard.api.motion_goals.allow_all_collisions()
        default_pose_giskard.execute(
            expected_error_type=EmptyProblemException, local_min_end=False
        )
        current_state = {
            k.name: v[0] for k, v in default_pose_giskard.api.world.state.items()
        }
        default_pose_giskard.compare_joint_state(
            current_state, default_pose_giskard.default_pose
        )

    def test_self_collision_avoidance(self, default_pose_giskard: HSRTester):
        r_goal = PoseStamped()
        r_goal.header.frame_id = default_pose_giskard.tip
        r_goal.pose.position.z = 0.5
        r_goal.pose.orientation.w = 1.0
        default_pose_giskard.api.motion_goals.add_cartesian_pose(
            goal_pose=r_goal, tip_link=default_pose_giskard.tip, root_link="map"
        )
        default_pose_giskard.execute()

    def test_self_collision_avoidance2(self, default_pose_giskard: HSRTester):
        js = {
            "arm_flex_joint": 0.0,
            "arm_lift_joint": 0.0,
            "arm_roll_joint": -1.52,
            "head_pan_joint": -0.09,
            "head_tilt_joint": -0.62,
            "wrist_flex_joint": -1.55,
            "wrist_roll_joint": 0.11,
        }
        default_pose_giskard.api.monitors.add_set_seed_configuration(js)
        default_pose_giskard.api.motion_goals.allow_all_collisions()
        default_pose_giskard.execute()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "hand_palm_link"
        goal_pose.pose.position.x = 0.5
        goal_pose.pose.orientation.w = 1.0
        default_pose_giskard.api.motion_goals.add_cartesian_pose(
            goal_pose=goal_pose, tip_link=default_pose_giskard.tip, root_link="map"
        )
        default_pose_giskard.execute()

    def test_attached_collision1(self, box_setup: HSRTester):
        box_name = "asdf"
        box_pose = PoseStamped()
        box_pose.header.frame_id = "map"
        box_pose.pose.position = Point(x=0.85, y=0.3, z=0.66)
        box_pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        box_setup.add_box_to_world(box_name, (0.07, 0.04, 0.1), box_pose)
        box_setup.open_gripper()

        grasp_pose = deepcopy(box_pose)
        # grasp_pose.pose.position.x -= 0.05
        q = quaternion_from_rotation_matrix(
            [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
        )
        grasp_pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        grasp = box_setup.api.motion_goals.add_motion_goal(
            class_name=GraspSequence.__name__,
            name="pick up",
            tip_link=box_setup.tip,
            root_link="map",
            gripper_joint="hand_motor_joint",
            goal_pose=grasp_pose,
        )
        detected = box_setup.api.monitors.add_pulse(name="Detect Object", after_ticks=5)
        success = box_setup.api.monitors.add_time_above(
            name="Obj in Hand?", threshold=10
        )
        stop_retry = box_setup.api.monitors.add_pulse(
            name="Above 5 Retries", after_ticks=100000
        )

        not_obj_in_hand = f"not {success}"
        box_setup.api.update_end_condition(node_name=detected, condition=detected)
        box_setup.api.update_reset_condition(
            node_name=detected, condition=not_obj_in_hand
        )

        box_setup.api.update_start_condition(node_name=grasp, condition=detected)
        box_setup.api.update_end_condition(node_name=grasp, condition=grasp)
        box_setup.api.update_reset_condition(node_name=grasp, condition=not_obj_in_hand)

        box_setup.api.update_start_condition(node_name=success, condition=grasp)
        box_setup.api.update_end_condition(node_name=success, condition=success)
        box_setup.api.update_reset_condition(
            node_name=success, condition=not_obj_in_hand
        )

        box_setup.api.update_start_condition(
            node_name=stop_retry, condition=f"{grasp} and not {success}"
        )
        box_setup.api.update_reset_condition(
            node_name=stop_retry, condition=f"not {stop_retry}"
        )

        box_setup.api.monitors.add_end_motion(start_condition=success)
        box_setup.api.monitors.add_cancel_motion(
            start_condition=stop_retry, error=Exception("too many retries")
        )
        box_setup.api.motion_goals.allow_all_collisions()
        box_setup.execute(local_min_end=False)
        box_setup.update_parent_link_of_group(box_name, box_setup.tip)

        base_goal = PoseStamped()
        base_goal.header.frame_id = box_setup.default_root
        base_goal.pose.position.x -= 0.5
        base_goal.pose.orientation.w = 1.0
        box_setup.move_base(base_goal)

    def test_schnibbeln_sequence(self, box_setup: HSRTester):
        box_name = "Schnibbler"
        box_pose = PoseStamped()
        box_pose.header.frame_id = box_setup.tip
        box_pose.pose.position = Point(x=0.0, y=0.0, z=0.06)
        box_pose.pose.orientation.w = 1.0
        bread_name = "Bernd"
        bread_pose = PoseStamped()
        bread_pose.header.frame_id = "map"
        bread_pose.pose.position = Point(x=0.91, y=0.25, z=0.62)
        bread_pose.pose.orientation.w = 1.0

        box_setup.add_box_to_world(
            name=box_name,
            size=(0.05, 0.01, 0.15),
            pose=box_pose,
            parent_link=box_setup.tip,
        )
        box_setup.add_box_to_world(
            name=bread_name, size=(0.1, 0.2, 0.06), pose=bread_pose, parent_link="box"
        )
        # box_setup.dye_group(group_name=box_name, rgba=(0.0, 0.588, 0.784, 1.0))
        # box_setup.dye_group(group_name=bread_name, rgba=(0.784, 0.588, 0.0, 1.0))
        box_setup.close_gripper()

        pre_schnibble_pose = PoseStamped()
        pre_schnibble_pose.header.frame_id = "map"
        pre_schnibble_pose.pose.position = Point(x=0.85, y=0.2, z=0.75)
        q = quaternion_from_rotation_matrix(
            [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
        )
        pre_schnibble_pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        pre_schnibble = "Position Knife"
        box_setup.api.motion_goals.add_cartesian_pose(
            name=pre_schnibble,
            goal_pose=pre_schnibble_pose,
            tip_link=box_setup.tip,
            root_link="map",
            end_condition=pre_schnibble,
        )
        human_close = box_setup.api.monitors.add_pulse(
            name="Human Close?",
            after_ticks=50,
            true_for_ticks=50,
            start_condition=pre_schnibble,
            end_condition="",
        )

        cut = box_setup.api.motion_goals.add_motion_goal(
            class_name=Cutting.__name__,
            name="Cut",
            root_link="map",
            tip_link=box_name,
            depth=0.1,
            right_shift=-0.1,
            start_condition=pre_schnibble,
        )

        # no_contact = box_setup.api.monitors.add_const_true(name='Made Contact?',
        #                                                start_condition=schnibble_down)

        schnibbel_done = box_setup.api.monitors.add_time_above(
            name="Done?", threshold=5, start_condition=cut
        )

        reset = f"not {schnibbel_done}"
        box_setup.api.update_reset_condition(node_name=cut, condition=reset)
        box_setup.api.update_reset_condition(node_name=schnibbel_done, condition=reset)
        box_setup.api.update_end_condition(
            node_name=human_close, condition=schnibbel_done
        )

        box_setup.api.update_pause_condition(node_name=cut, condition=human_close)

        box_setup.api.monitors.add_end_motion(start_condition=schnibbel_done)
        # box_setup.api.monitors.add_cancel_motion(start_condition=f'not {no_contact}', error=Exception('no contact'))
        box_setup.api.motion_goals.allow_all_collisions()
        box_setup.execute(local_min_end=False)
        # box_setup.update_parent_link_of_group(box_name, box_setup.tip)

    def test_collision_avoidance(self, default_pose_giskard: HSRTester):
        js = {"arm_flex_joint": -np.pi / 2}
        default_pose_giskard.api.motion_goals.add_joint_position(js)
        default_pose_giskard.execute()

        p = PoseStamped()
        p.header.frame_id = "map"
        p.pose.position.x = 0.9
        p.pose.position.y = 0.0
        p.pose.position.z = 0.5
        p.pose.orientation.w = 1.0
        default_pose_giskard.add_box_to_world(name="box", size=(1, 1, 0.01), pose=p)

        js = {"arm_flex_joint": 0.0}
        default_pose_giskard.api.motion_goals.add_joint_position(js)
        default_pose_giskard.execute()

    #
    # def test_avoid_collision_touch_hard_threshold(self, box_setup: HSRTestWrapper):
    #     base_goal = PoseStamped()
    #     base_goal.header.frame_id = box_setup.default_root
    #     base_goal.pose.position.x = 0.2
    #     base_goal.pose.orientation.z = 1
    #     box_setup.teleport_base(base_goal)
    #
    #     box_setup.avoid_collision(min_distance=0.05, group1=box_setup.robot_name)
    #     box_setup.allow_self_collision()
    #
    #     base_goal = PoseStamped()
    #     base_goal.header.frame_id = 'base_footprint'
    #     base_goal.pose.position.x = -0.3
    #     base_goal.pose.orientation.w = 1
    #     box_setup.api.motion_goals.add_joint_position(base_goal, tip_link='base_footprint', root_link='map', weight=WEIGHT_ABOVE_CA)
    #     box_setup.set_max_traj_length(30)
    #     box_setup.execute(local_min_end=False)
    #     box_setup.check_cpi_geq(['base_link'], 0.048)
    #     box_setup.check_cpi_leq(['base_link'], 0.07)


class TestAddObject:
    def test_add(self, default_pose_giskard):
        box1_name = "box1"
        pose = PoseStamped()
        pose.header.frame_id = default_pose_giskard.default_root
        pose.pose.orientation.w = 1.0
        pose.pose.position.x = 1.0
        default_pose_giskard.add_box_to_world(
            name=box1_name, size=(1, 1, 1), pose=pose, parent_link="hand_palm_link"
        )

        default_pose_giskard.api.motion_goals.add_joint_position(
            {"arm_flex_joint": -0.7}
        )
        default_pose_giskard.execute()
