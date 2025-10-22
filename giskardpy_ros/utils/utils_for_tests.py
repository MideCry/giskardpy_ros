import asyncio
import csv
import os
from copy import deepcopy
from threading import Thread
from time import time, sleep
from typing import Tuple, Optional, List, Dict, Union, Iterable

import giskard_msgs.msg as giskard_msgs
import numpy as np
from angles import shortest_angular_distance
from geometry_msgs.msg import PoseStamped, Point, PointStamped, Quaternion, Pose
from giskard_msgs.action._move import Move_Result, Move_Goal
from giskard_msgs.msg import GiskardError
from giskard_msgs.srv._dye_group import DyeGroup_Response
from rclpy.publisher import Publisher
from sensor_msgs.msg import JointState
from tf2_py import LookupException, ExtrapolationException

import giskardpy_ros.ros2.msg_converter as msg_converter
import giskardpy_ros.ros2.tfwrapper as tf
import semantic_world.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import (
    UnknownGroupException,
    WorldException,
)
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.model.collision_matrix_manager import CollisionViewRequest
from giskardpy.model.collisions import Collisions, GiskardCollision
from giskardpy.motion_statechart.tasks.diff_drive_goals import (
    DiffDriveTangentialToPoint,
    KeepHandInWorkspace,
)
from giskardpy.motion_statechart.tasks.task import WEIGHT_ABOVE_CA
from giskardpy.qp.solvers.qp_solver_ids import SupportedQPSolver
from giskardpy_ros.configs.giskard import Giskard
from giskardpy_ros.python_interface.python_interface import GiskardWrapperNode
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard
from giskardpy_ros.utils.utils import is_in_github_workflow
from semantic_world.adapters.urdf import URDFParser
from semantic_world.datastructures.prefixed_name import PrefixedName
from semantic_world.robots.abstract_robot import AbstractRobot
from semantic_world.spatial_types.derivatives import Derivatives
from semantic_world.world_description.connections import (
    OmniDrive,
    PrismaticConnection,
    RevoluteConnection,
    FixedConnection,
    ActiveConnection1DOF,
)
from semantic_world.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_world.world_description.geometry import (
    Box,
    Scale,
    Sphere,
    Cylinder,
    FileMesh,
)
from semantic_world.world_description.world_entity import Body


def compare_poses(
    actual_pose: Union[cas.TransformationMatrix, Pose],
    desired_pose: Union[cas.TransformationMatrix, Pose],
    decimal: int = 2,
) -> None:
    if isinstance(actual_pose, cas.TransformationMatrix):
        actual_pose = msg_converter.to_ros_message(actual_pose).pose
    if isinstance(desired_pose, cas.TransformationMatrix):
        desired_pose = msg_converter.to_ros_message(desired_pose).pose
    compare_points(
        actual_point=actual_pose.position,
        desired_point=desired_pose.position,
        decimal=decimal,
    )
    compare_orientations(
        actual_orientation=actual_pose.orientation,
        desired_orientation=desired_pose.orientation,
        decimal=decimal,
    )


def compare_points(
    actual_point: Union[cas.Point3, Point],
    desired_point: Union[cas.Point3, Point],
    decimal: int = 2,
) -> None:
    if isinstance(actual_point, cas.Point3):
        actual_point = msg_converter.to_ros_message(actual_point).point
    if isinstance(desired_point, cas.Point3):
        desired_point = msg_converter.to_ros_message(desired_point).point
    np.testing.assert_almost_equal(actual_point.x, desired_point.x, decimal=decimal)
    np.testing.assert_almost_equal(actual_point.y, desired_point.y, decimal=decimal)
    np.testing.assert_almost_equal(actual_point.z, desired_point.z, decimal=decimal)


def compare_orientations(
    actual_orientation: Union[Quaternion, np.ndarray],
    desired_orientation: Union[Quaternion, np.ndarray],
    decimal: int = 2,
) -> None:
    if isinstance(actual_orientation, Quaternion):
        q1 = np.array(
            [
                actual_orientation.x,
                actual_orientation.y,
                actual_orientation.z,
                actual_orientation.w,
            ]
        )
    else:
        q1 = actual_orientation
    if isinstance(desired_orientation, Quaternion):
        q2 = np.array(
            [
                desired_orientation.x,
                desired_orientation.y,
                desired_orientation.z,
                desired_orientation.w,
            ]
        )
    else:
        q2 = desired_orientation
    try:
        np.testing.assert_almost_equal(q1[0], q2[0], decimal=decimal)
        np.testing.assert_almost_equal(q1[1], q2[1], decimal=decimal)
        np.testing.assert_almost_equal(q1[2], q2[2], decimal=decimal)
        np.testing.assert_almost_equal(q1[3], q2[3], decimal=decimal)
    except:
        np.testing.assert_almost_equal(q1[0], -q2[0], decimal=decimal)
        np.testing.assert_almost_equal(q1[1], -q2[1], decimal=decimal)
        np.testing.assert_almost_equal(q1[2], -q2[2], decimal=decimal)
        np.testing.assert_almost_equal(q1[3], -q2[3], decimal=decimal)


def position_dict_to_joint_states(joint_state_dict: Dict[str, float]) -> JointState:
    """
    :param joint_state_dict: maps joint_name to position
    :return: velocity and effort are filled with 0
    """
    js = JointState()
    for k, v in joint_state_dict.items():
        js.name.append(k)
        js.position.append(v)
        js.velocity.append(0)
        js.effort.append(0)
    return js


class GiskardTester:
    default_pose: Dict[str, float]
    better_pose = Dict[str, float]
    odom_root = "odom"
    api: GiskardWrapperNode
    giskard: Giskard

    def __init__(self, giskard: Giskard):
        self.async_loop = asyncio.new_event_loop()
        self.total_time_spend_giskarding = 0
        self.total_time_spend_moving = 0
        self.default_env_name: Optional[str] = None
        self.env_joint_state_pubs: Dict[str, Publisher] = {}

        self.giskard = giskard
        self.giskard.setup()
        if is_in_github_workflow():
            get_middleware().loginfo(
                "Inside github workflow, turning off visualization"
            )
            GiskardBlackboard().tree.turn_off_visualization()
        if "QP_SOLVER" in os.environ:
            god_map.qp_controller.set_qp_solver(
                SupportedQPSolver[os.environ["QP_SOLVER"]]
            )
        self.robot_names = [
            v.name for v in god_map.world.get_views_by_type(AbstractRobot)
        ]
        self.default_root = str(god_map.world.root.name.name)

        # rospy.sleep(1)
        self.original_number_of_links = len(god_map.world.bodies)
        self.heart = Thread(target=GiskardBlackboard().tree.live, name="bt ticker")
        self.heart.start()
        self.wait_heartbeats(1)
        self.api = GiskardWrapperNode(node_name="tests")

    def get_odometry_joint(self) -> OmniDrive:
        return god_map.world.get_views_by_type(AbstractRobot)[0].drive

    def compute_fk_pose(self, root_link: str, tip_link: str) -> PoseStamped:
        root_T_tip = god_map.world.compute_forward_kinematics(
            root=god_map.world.get_kinematic_structure_entity_by_name(root_link),
            tip=god_map.world.get_kinematic_structure_entity_by_name(tip_link),
        )
        return msg_converter.to_ros_message(root_T_tip)

    def compute_fk_point(self, root_link: str, tip_link: str) -> PointStamped:
        root_T_tip = god_map.world.compute_forward_kinematics(
            root=god_map.world.get_kinematic_structure_entity_by_name(root_link),
            tip=god_map.world.get_kinematic_structure_entity_by_name(tip_link),
        ).to_position()
        return msg_converter.to_ros_message(root_T_tip)

    def has_odometry_joint(self) -> bool:
        try:
            joint = self.get_odometry_joint()
        except WorldException as e:
            return False
        return isinstance(joint, (OmniDrive,))

    def set_seed_odometry(self, base_pose, group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.api.robot_name
        self.api.monitors.add_set_seed_configuration(
            group_name=group_name, base_pose=base_pose
        )

    def transform_msg(self, target_frame, msg, timeout=1):
        result_msg = deepcopy(msg)
        try:
            if not GiskardBlackboard().tree_config.is_standalone():
                return tf.transform_msg(target_frame, result_msg, timeout=timeout)
            else:
                raise LookupException("just to trigger except block")
        except (LookupException, ExtrapolationException) as e:
            target_frame = god_map.world.get_kinematic_structure_entity_by_name(
                target_frame
            ).name
            try:
                result_msg.header.frame_id = str(
                    god_map.world.get_kinematic_structure_entity_by_name(
                        result_msg.header.frame_id
                    ).name.name
                )
            except UnknownGroupException:
                pass
            giskard_obj = msg_converter.ros_msg_to_giskard_obj(
                result_msg, god_map.world
            )
            target_body = god_map.world.get_kinematic_structure_entity_by_name(
                target_frame
            )
            transformed_giskard_obj = god_map.world.transform(
                target_frame=target_body, spatial_object=giskard_obj
            )
            return msg_converter.to_ros_message(transformed_giskard_obj)

    def wait_heartbeats(self, number=5):
        behavior_tree = GiskardBlackboard().tree
        c = behavior_tree.count
        while behavior_tree.count < c + number:
            sleep(0.001)

    def dye_group(
        self,
        group_name: str,
        rgba: Tuple[float, float, float, float],
        expected_error_codes=(DyeGroup_Response.SUCCESS,),
    ):
        pass

    def print_qp_solver_times(self):
        file_name = f"{god_map.tmp_folder}/benchmark.csv"
        with open(file_name, mode="w", newline="") as csvfile:
            csvwriter = csv.writer(
                csvfile, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csvwriter.writerow(
                [
                    "solver",
                    "filtered_variables",
                    "variables",
                    "eq_constraints",
                    "neq_constraints",
                    "num_eq_slack_variables",
                    "num_neq_slack_variables",
                    "num_slack_variables",
                    "max_derivative",
                    "data",
                ]
            )

            # for solver_id, solver_class in available_solvers.items():
            #     times = solver_class.get_solver_times()
            #     for (filtered_variables, variables, eq_constraints, neq_constraints, num_eq_slack_variables,
            #          num_neq_slack_variables, num_slack_variables), times in sorted(times.items()):
            #         csvwriter.writerow([solver_id.name,
            #                             str(filtered_variables),
            #                             str(variables),
            #                             str(eq_constraints),
            #                             str(neq_constraints),
            #                             str(num_eq_slack_variables),
            #                             str(num_neq_slack_variables),
            #                             str(num_slack_variables),
            #                             str(int(god_map.qp_controller.max_derivative)),
            #                             str(times)])

        get_middleware().loginfo(f"saved benchmark file in {file_name}")

    def print_stats(self):
        giskarding_time = self.total_time_spend_giskarding
        if not GiskardBlackboard().tree_config.is_standalone():
            giskarding_time -= self.total_time_spend_moving
        get_middleware().loginfo(f"total time spend giskarding: {giskarding_time}")
        get_middleware().loginfo(
            f"total time spend moving: {self.total_time_spend_moving}"
        )

    def set_env_state(self, joint_state: Dict[str, float]):
        self.api.monitors.add_set_seed_configuration(
            seed_configuration=joint_state, name="set kitchen state"
        )
        self.api.motion_goals.allow_all_collisions()
        self.execute()
        self.wait_heartbeats()

    def compare_joint_state(
        self,
        current_js: Dict[Union[str, PrefixedName], float],
        goal_js: Dict[Union[str, PrefixedName], float],
        decimal: int = 2,
    ):
        for joint_name in goal_js:
            goal = goal_js[joint_name]
            current = current_js[joint_name]
            connection: ActiveConnection1DOF = god_map.world.get_connection_by_name(
                joint_name
            )
            if not connection.dof.has_position_limits():
                np.testing.assert_almost_equal(
                    shortest_angular_distance(goal, current),
                    0,
                    decimal=decimal,
                    err_msg=f"{joint_name}: actual: {current} desired: {goal}",
                )
            else:
                np.testing.assert_almost_equal(
                    current,
                    goal,
                    decimal,
                    err_msg=f"{joint_name}: actual: {current} desired: {goal}",
                )

    #
    # GOAL STUFF #################################################################################################
    #

    def teleport_base(self, goal_pose, group_name: Optional[str] = None):
        done = self.api.monitors.add_set_seed_odometry(
            base_pose=goal_pose, group_name=group_name, name="teleport base"
        )
        self.api.motion_goals.allow_all_collisions()
        self.api.monitors.add_end_motion(start_condition=done)
        self.execute()

    def set_keep_hand_in_workspace(
        self,
        tip_link: Union[str, giskard_msgs.LinkName],
        map_frame=None,
        base_footprint=None,
    ):
        if isinstance(tip_link, str):
            tip_link = giskard_msgs.LinkName(name=tip_link)
        self.api.motion_goals.add_motion_goal(
            class_name=KeepHandInWorkspace.__name__,
            tip_link=tip_link,
            map_frame=map_frame,
            base_footprint=base_footprint,
        )

    def set_diff_drive_tangential_to_point(
        self, goal_point: PointStamped, weight: float = WEIGHT_ABOVE_CA, **kwargs
    ):
        self.api.motion_goals.add_motion_goal(
            class_name=DiffDriveTangentialToPoint.__name__,
            goal_point=goal_point,
            weight=weight,
            **kwargs,
        )

    #
    # GENERAL GOAL STUFF ###############################################################################################
    #

    def execute(
        self,
        expected_error_type: Optional[type(Exception)] = None,
        stop_after: float = None,
        wait: bool = True,
        local_min_end: bool = True,
    ) -> Move_Result:
        if local_min_end:
            self.api.add_default_end_motion_conditions()
        return self.async_loop.run_until_complete(
            self.send_goal(
                expected_error_type=expected_error_type,
                stop_after=stop_after,
                wait=wait,
            )
        )

    def projection(
        self,
        expected_error_type: Optional[type(Exception)] = None,
        wait: bool = True,
        add_local_minimum_reached: bool = True,
    ) -> Move_Result:
        """
        Plans, but doesn't execute the goal. Useful, if you just want to look at the planning ghost.
        :param wait: this function blocks if wait=True
        :return: result from Giskard
        """
        if add_local_minimum_reached:
            self.api.add_default_end_motion_conditions()
        last_js = god_map.world.state.to_position_dict()
        for key, value in list(last_js.items()):
            if key not in god_map.world.controlled_connections:
                del last_js[key]
        result = self.async_loop.run_until_complete(
            self.send_goal(
                expected_error_type=expected_error_type,
                goal_type=Move_Goal.PROJECTION,
                wait=wait,
            )
        )
        new_js = god_map.world.state.to_position_dict()
        for key, value in list(new_js.items()):
            if key not in god_map.world.controlled_connections:
                del new_js[key]
        self.compare_joint_state(new_js, last_js)
        return result

    def plan(
        self,
        expected_error_type: Optional[type(Exception)] = None,
        wait: bool = True,
        add_local_minimum_reached: bool = True,
    ) -> Move_Result:
        return self.projection(
            expected_error_type=expected_error_type,
            wait=wait,
            add_local_minimum_reached=add_local_minimum_reached,
        )

    async def send_goal(
        self,
        expected_error_type: Optional[type(Exception)] = None,
        goal_type: int = Move_Goal.EXECUTE,
        goal: Optional[Move_Goal] = None,
        stop_after: Optional[float] = None,
        wait: bool = True,
    ) -> Optional[Move_Result]:
        try:
            time_spend_giskarding = time()
            future_goal_accepted = self.api._send_action_goal_async(
                goal_type
            )  # FIXME set breakpoint here to avoid long waiting times in debug mode
            await future_goal_accepted
            if stop_after is not None:
                await asyncio.sleep(stop_after)
                cancel_result = await self.api.cancel_goal_async()
                # assert cancel_result.
                r = await self.api.get_result()
            elif wait:
                r = await self.api.get_result()
            else:
                return
            # self.wait_heartbeats()
            diff = time() - time_spend_giskarding
            self.total_time_spend_giskarding += diff
            self.total_time_spend_moving += (
                len(god_map.trajectory) * god_map.qp_controller.config.mpc_dt
            )
            get_middleware().logwarn(f"Goal processing took {diff}")
            result_exception = msg_converter.error_msg_to_exception(r.error)
            if expected_error_type is not None:
                assert type(result_exception) == expected_error_type, (
                    f"got: {type(result_exception)}, "
                    f"expected: {expected_error_type} | error_massage: {r.error.msg}"
                )
            else:
                if result_exception is not None:
                    raise result_exception
            # self.are_joint_limits_violated()
        finally:
            self.sync_world_with_trajectory()
        return r

    def sync_world_with_trajectory(self):
        t = god_map.trajectory
        whole_last_joint_state = t[-1].to_position_dict()
        for group_name in self.env_joint_state_pubs:
            group_joints = self.api.world.get_group_info(group_name).joint_state.name
            group_last_joint_state = {
                str(k): v
                for k, v in whole_last_joint_state.items()
                if k in group_joints
            }
            self.set_env_state(group_last_joint_state, group_name)

    def get_result_trajectory_position(self):
        trajectory = god_map.trajectory
        trajectory2 = {}
        for joint_name in trajectory[0].keys():
            trajectory2[joint_name] = np.array(
                [p[joint_name].position for p in trajectory]
            )
        return trajectory2

    def get_result_trajectory_velocity(self):
        trajectory = god_map.trajectory
        trajectory2 = {}
        for joint_name in trajectory[0].keys():
            trajectory2[joint_name] = np.array(
                [p[joint_name].velocity for p in trajectory]
            )
        return trajectory2

    def are_joint_limits_violated(self, eps=1e-2):
        active_free_variables: List[DegreeOfFreedom] = (
            god_map.qp_controller.degrees_of_freedoms
        )
        for free_variable in active_free_variables:
            if free_variable.has_position_limits():
                lower_limit = free_variable.get_lower_limit(Derivatives.position)
                upper_limit = free_variable.get_upper_limit(Derivatives.position)
                if not isinstance(lower_limit, float):
                    lower_limit = lower_limit.to_np()
                if not isinstance(upper_limit, float):
                    upper_limit = upper_limit.to_np()
                current_position = god_map.world.state[free_variable.name].position
                assert (
                    lower_limit - eps <= current_position <= upper_limit + eps
                ), f"joint limit of {free_variable.name} is violated {lower_limit} <= {current_position} <= {upper_limit}"

    def are_joint_limits_in_traj_violated(self):
        trajectory_vel = self.get_result_trajectory_velocity()
        trajectory_pos = self.get_result_trajectory_position()
        controlled_joints = god_map.world.controlled_connections
        for joint_name in controlled_joints:
            if isinstance(
                god_map.world.joints[joint_name],
                (PrismaticConnection, RevoluteConnection),
            ):
                if not god_map.world.is_joint_continuous(joint_name):
                    joint_limits = god_map.world.get_joint_position_limits(joint_name)
                    error_msg = f"{joint_name} has violated joint position limit"
                    eps = 0.0001
                    np.testing.assert_array_less(
                        trajectory_pos[joint_name], joint_limits[1] + eps, error_msg
                    )
                    np.testing.assert_array_less(
                        -trajectory_pos[joint_name], -joint_limits[0] + eps, error_msg
                    )
                vel_limit = (
                    god_map.world.get_joint_velocity_limits(joint_name)[1] * 1.001
                )
                vel = trajectory_vel[joint_name]
                error_msg = f"{joint_name} has violated joint velocity limit {vel} > {vel_limit}"
                assert np.all(np.less_equal(vel, vel_limit)), error_msg
                assert np.all(np.greater_equal(vel, -vel_limit)), error_msg

    #
    # BULLET WORLD #####################################################################################################
    #

    def register_group(
        self, new_group_name: str, root_link_name: giskard_msgs.LinkName
    ):
        self.api.world.register_group(
            new_group_name=new_group_name, root_link_name=root_link_name
        )
        # self.wait_heartbeats()
        assert new_group_name in self.api.world.get_group_names()

    def remove_group(
        self, name: str, expected_error_type: Optional[type(Exception)] = None
    ) -> None:
        old_link_names = []
        old_joint_names = []
        if expected_error_type is None:
            old_link_names = god_map.world.groups[name].link_names_as_set
            old_joint_names = god_map.world.groups[name].joint_names
        try:
            r = self.api.world.remove_group(name)
            self.wait_heartbeats()
            assert r.error.type == GiskardError.SUCCESS
            # links removed from world
            for old_link_name in old_link_names:
                assert old_link_name not in god_map.world.link_names_as_set
            # joints removed from world
            for old_joint_name in old_joint_names:
                assert old_joint_name not in god_map.world.joint_names
            # links removed from collision scene
            for link_a, link_b in god_map.collision_scene.self_collision_matrix:
                try:
                    assert link_a not in old_link_names
                    assert link_b not in old_link_names
                except AssertionError as e:
                    pass
            return r
        except Exception as e:
            assert type(e) == expected_error_type
        assert name not in god_map.world.groups
        assert name not in self.api.world.get_group_names()
        # if name in self.env_joint_state_pubs: todo
        #     self.env_joint_state_pubs[name].unregister()
        #     del self.env_joint_state_pubs[name]
        if name == self.default_env_name:
            self.default_env_name = None

    def detach_group(
        self, name: str, expected_error_type: Optional[type(Exception)] = None
    ) -> None:
        with self.api.world.modify_world():
            body = self.api.world.get_body_by_name(name)
            parent_T_connection = self.api.world.compute_forward_kinematics(
                self.api.world.root, body
            )
            new_connection = FixedConnection(
                parent=self.api.world.root,
                child=body,
                parent_T_connection_expression=parent_T_connection,
            )
            self.api.world.remove_connection(body.parent_connection)
            self.api.world.add_connection(new_connection)
        self.wait_heartbeats()

    def check_add_object_result(
        self,
        name: str,
        pose: Optional[PoseStamped],
        parent_body_name: Optional[Union[str, giskard_msgs.LinkName]] = None,
        expected_error_type: Optional[type(Exception)] = None,
    ):
        pass  # fixme

    def add_box_to_world(
        self,
        name: str,
        size: Tuple[float, float, float],
        pose: PoseStamped,
        parent_link: Optional[Union[str, giskard_msgs.LinkName]] = None,
        expected_error_type: Optional[type(Exception)] = None,
    ) -> None:
        if parent_link is None:
            parent_link = self.api.world.root
        else:
            parent_link = self.api.world.get_kinematic_structure_entity_by_name(
                parent_link
            )
        parent_T_pose = self.api.world.transform(
            spatial_object=msg_converter.ros_msg_to_giskard_obj(pose, self.api.world),
            target_frame=parent_link,
        )
        with self.api.world.modify_world():
            box = Body(name=PrefixedName(name), _world=self.api.world)
            box_shape = Box(scale=Scale(*size))
            box.collision.append(box_shape)
            box.visual.append(box_shape)
            box.collision_config.buffer_zone_distance = 0.05

            connection = FixedConnection(
                parent=parent_link,
                child=box,
                parent_T_connection_expression=parent_T_pose,
            )
            self.api.world.add_connection(connection)
            self.api.world.add_body(box)
        self.wait_heartbeats()
        self.check_add_object_result(
            name=name,
            pose=parent_T_pose,
            parent_body_name=parent_link,
            expected_error_type=expected_error_type,
        )

    def update_group_pose(
        self,
        group_name: str,
        new_pose: PoseStamped,
        expected_error_type: Optional[type(Exception)] = None,
    ) -> None:
        try:
            response = self.api.world.update_group_pose(
                group_name=group_name, new_pose=new_pose
            )
            self.wait_heartbeats()
            assert response.error.type == GiskardError.SUCCESS
            info = self.api.world.get_group_info(group_name)
            map_T_group = tf.transform_pose(god_map.world.root.name, new_pose)
            compare_poses(info.root_link_pose.pose, map_T_group.pose)
        except Exception as e:
            assert type(e) == expected_error_type

    def add_sphere_to_world(
        self,
        name: str,
        radius: float = 1.0,
        pose: PoseStamped = None,
        parent_link: Optional[Union[str, giskard_msgs.LinkName]] = None,
        expected_error_type: Optional[type(Exception)] = None,
    ) -> None:
        if parent_link is None:
            parent_link = self.api.world.root
        else:
            parent_link = self.api.world.get_kinematic_structure_entity_by_name(
                parent_link
            )
        with self.api.world.modify_world():
            sphere = Body(name=PrefixedName(name), _world=self.api.world)
            sphere_shape = Sphere(radius=radius)
            sphere.collision.append(sphere_shape)
            sphere.visual.append(sphere_shape)

            connection = FixedConnection(
                parent=parent_link,
                child=sphere,
                connection_T_child_expression=msg_converter.ros_msg_to_giskard_obj(
                    pose, self.api.world
                ),
            )
            self.api.world.add_connection(connection)
            self.api.world.add_body(sphere)
        self.wait_heartbeats()
        self.check_add_object_result(
            name=name,
            pose=pose,
            parent_body_name=parent_link,
            expected_error_type=expected_error_type,
        )

    def add_cylinder_to_world(
        self,
        name: str,
        height: float,
        radius: float,
        pose: PoseStamped = None,
        parent_link: Optional[Union[str, giskard_msgs.LinkName]] = None,
        expected_error_type: Optional[type(Exception)] = None,
    ) -> None:
        if parent_link is None:
            parent_link = self.api.world.root
        else:
            parent_link = self.api.world.get_kinematic_structure_entity_by_name(
                parent_link
            )
        parent_T_pose = self.api.world.transform(
            spatial_object=msg_converter.ros_msg_to_giskard_obj(pose, self.api.world),
            target_frame=parent_link,
        )
        with self.api.world.modify_world():
            cylinder = Body(name=PrefixedName(name), _world=self.api.world)
            cylinder_shape = Cylinder(width=radius * 2, height=height)
            cylinder.collision.append(cylinder_shape)
            cylinder.visual.append(cylinder_shape)
            cylinder.collision_config.buffer_zone_distance = 0.05

            connection = FixedConnection(
                parent=parent_link,
                child=cylinder,
                parent_T_connection_expression=parent_T_pose,
            )
            self.api.world.add_connection(connection)
            self.api.world.add_body(cylinder)
        self.wait_heartbeats()
        self.check_add_object_result(
            name=name,
            pose=parent_T_pose,
            parent_body_name=parent_link,
            expected_error_type=expected_error_type,
        )

    def add_mesh_to_world(
        self,
        pose: PoseStamped,
        name: str = "meshy",
        mesh: str = "",
        parent_link: Optional[Union[str, giskard_msgs.LinkName]] = None,
        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        expected_error_type: Optional[type(Exception)] = None,
    ) -> None:
        if parent_link is None:
            parent_link = self.api.world.root
        else:
            parent_link = self.api.world.get_kinematic_structure_entity_by_name(
                parent_link
            )
        parent_T_pose = self.api.world.transform(
            spatial_object=msg_converter.ros_msg_to_giskard_obj(pose, self.api.world),
            target_frame=parent_link,
        )
        with self.api.world.modify_world():
            mesh_body = Body(name=PrefixedName(name), _world=self.api.world)
            mesh_shape = FileMesh(filename=mesh, scale=Scale(*scale))
            mesh_body.collision.append(mesh_shape)
            mesh_body.visual.append(mesh_shape)
            mesh_body.collision_config.buffer_zone_distance = 0.05

            connection = FixedConnection(
                parent=parent_link,
                child=mesh_body,
                parent_T_connection_expression=parent_T_pose,
            )
            self.api.world.add_connection(connection)
            self.api.world.add_body(mesh_body)
        self.wait_heartbeats()
        self.check_add_object_result(
            name=name,
            pose=parent_T_pose,
            parent_body_name=parent_link,
            expected_error_type=expected_error_type,
        )

    def add_urdf_to_world(
        self,
        name: str,
        urdf: str,
        pose: PoseStamped,
        parent_link: Optional[Union[str, giskard_msgs.LinkName]] = None,
        js_topic: Optional[str] = "",
        set_js_topic: Optional[str] = "",
        expected_error_type: Optional[type(Exception)] = None,
    ) -> None:
        if parent_link is None:
            parent_link = self.api.world.root
        else:
            parent_link = self.api.world.get_kinematic_structure_entity_by_name(
                parent_link
            )
        pr2_parser = URDFParser(urdf=urdf, prefix=name)
        world_with_pr2 = pr2_parser.parse()
        with self.api.world.modify_world():
            c_map_root = FixedConnection(
                parent=parent_link,
                child=world_with_pr2.root,
                connection_T_child_expression=msg_converter.ros_msg_to_giskard_obj(
                    pose, self.api.world
                ),
            )
            self.api.world.merge_world(world_with_pr2, root_connection=c_map_root)

        self.wait_heartbeats()

    def update_parent_link_of_group(
        self,
        name: str,
        parent_link: Optional[Union[str, giskard_msgs.LinkName]] = None,
        expected_error_type: Optional[type(Exception)] = None,
    ) -> None:
        with self.api.world.modify_world():
            body = self.api.world.get_kinematic_structure_entity_by_name(name)
            parent = self.api.world.get_kinematic_structure_entity_by_name(parent_link)
            parent_T_connection = self.api.world.compute_forward_kinematics(
                parent, body
            )
            new_connection = FixedConnection(
                parent=parent,
                child=body,
                parent_T_connection_expression=parent_T_connection,
            )
            self.api.world.remove_connection(body.parent_connection)
            self.api.world.add_connection(new_connection)
        self.wait_heartbeats()

    def get_external_collisions(self) -> Collisions:
        collision_goals = []
        for robot_name in self.robot_names:
            collision_goals.append(
                CollisionViewRequest(
                    type_=CollisionViewRequest.AVOID_COLLISION,
                    distance=None,
                    view1=robot_name,
                )
            )
            collision_goals.append(
                CollisionViewRequest(
                    type_=CollisionViewRequest.ALLOW_COLLISION,
                    distance=None,
                    view1=robot_name,
                    view2=robot_name,
                )
            )
        return self.compute_collisions(collision_goals)

    def get_self_collisions(self, group_name: Optional[str] = None) -> Collisions:
        if group_name is None:
            group_name = self.robot_names[0]
        collision_entries = [
            CollisionViewRequest(
                type_=CollisionViewRequest.AVOID_COLLISION,
                distance=None,
                view1=group_name,
                view2=group_name,
            )
        ]
        return self.compute_collisions(collision_entries)

    def compute_collisions(
        self, collision_entries: List[CollisionViewRequest]
    ) -> Collisions:
        god_map.collision_scene.collision_detector.reset_cache()
        god_map.collision_scene.matrix_manager.parse_collision_requests(
            collision_entries
        )
        collision_matrix = (
            god_map.collision_scene.matrix_manager.compute_collision_matrix()
        )
        god_map.collision_scene.set_collision_matrix(collision_matrix)
        return god_map.collision_scene.check_collisions()

    def compute_all_collisions(self) -> Collisions:
        collision_entries = [
            CollisionViewRequest(
                type_=CollisionViewRequest.AVOID_COLLISION, distance=None
            )
        ]
        return self.compute_collisions(collision_entries)

    def check_cpi_geq(
        self,
        bodies: Iterable[Body],
        distance_threshold: float,
        check_external: bool = True,
        check_self: bool = True,
    ):
        collisions = self.compute_all_collisions()
        for collision in collisions.all_collisions:
            if not check_external and collision.is_external:
                continue
            if not check_self and not collision.is_external:
                continue
            if (
                collision.original_body_a in bodies
                or collision.original_body_b in bodies
            ):
                assert collision.contact_distance >= distance_threshold, (
                    f"{collision.contact_distance} < {distance_threshold} "
                    f"({collision.original_body_a} with {collision.original_body_b})"
                )

    def check_cpi_leq(
        self,
        bodies: Iterable[Body],
        distance_threshold: float,
        check_external: bool = True,
        check_self: bool = True,
    ):
        collisions = self.compute_all_collisions()
        min_contact: GiskardCollision = None
        for collision in collisions.all_collisions:
            if not check_external and collision.is_external:
                continue
            if not check_self and not collision.is_external:
                continue
            if (
                collision.original_body_a in bodies
                or collision.original_body_b in bodies
            ):
                if (
                    min_contact is None
                    or collision.contact_distance <= min_contact.contact_distance
                ):
                    min_contact = collision
        assert min_contact.contact_distance <= distance_threshold, (
            f"{min_contact.contact_distance} > {distance_threshold} "
            f"({min_contact.original_body_a} with {min_contact.original_body_b})"
        )

    def move_base(self, goal_pose) -> None:
        tip = self.get_odometry_joint().child
        self.api.motion_goals.add_cartesian_pose(
            goal_pose=goal_pose,
            tip_link=tip.name,
            root_link="map",
            name="base goal",
        )
        self.execute()

    def reset_base(self):
        p = PoseStamped()
        p.header.frame_id = god_map.world.root.name
        p.pose.orientation.w = 1.0
        self.teleport_base(p)
