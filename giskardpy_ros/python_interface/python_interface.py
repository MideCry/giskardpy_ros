from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Thread
from time import sleep
from typing import Dict, Tuple, Optional, List, Union

import numpy as np
import rclpy
from geometry_msgs.msg import (
    PoseStamped,
    Vector3Stamped,
    PointStamped,
    QuaternionStamped,
)
from giskard_msgs.action import Move
from giskard_msgs.action._move import Move_Result
from giskard_msgs.msg import CollisionEntry, LinkName
from giskard_msgs.msg import ExecutionState, MotionStatechartNode
from nav_msgs.msg import Path
from rclpy import Context, Parameter, Future
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node

from giskardpy.data_types.exceptions import (
    MaxTrajectoryLengthException,
    ExecutionException,
)
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.data_types import goal_parameter
from giskardpy.motion_statechart.goals.align_to_push_door import AlignToPushDoor
from giskardpy.motion_statechart.goals.cartesian_goals import (
    DiffDriveBaseGoal,
    CartesianPoseStraight,
    CartesianPositionStraight,
)
from giskardpy.motion_statechart.goals.collision_avoidance import CollisionAvoidance
from giskardpy.motion_statechart.goals.open_close import Close, Open
from giskardpy.motion_statechart.goals.pre_push_door import PrePushDoor
from giskardpy.motion_statechart.monitors.cartesian_monitors import (
    PoseReached,
    PositionReached,
    OrientationReached,
    PointingAt,
    VectorsAligned,
    DistanceToLine,
)
from giskardpy.motion_statechart.monitors.feature_monitors import (
    PerpendicularMonitor,
    AngleMonitor,
    HeightMonitor,
    DistanceMonitor,
)
from giskardpy.motion_statechart.monitors.joint_monitors import JointGoalReached
from giskardpy.motion_statechart.monitors.monitors import (
    LocalMinimumReached,
    TimeAbove,
    Alternator,
    CancelMotion,
    EndMotion,
    TrueMonitor,
    FalseMonitor,
)
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetOdometry,
    SetSeedConfiguration,
)
from giskardpy.motion_statechart.monitors.payload_monitors import (
    Print,
    Sleep,
    Pulse,
    CheckMaxTrajectoryLength,
)
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPosition,
    CartesianOrientation,
    CartesianPose,
    JustinTorsoLimitCart,
    CartesianVelocityLimit,
)
from giskardpy.motion_statechart.tasks.feature_functions import (
    AlignPerpendicular,
    HeightGoal,
    AngleGoal,
    DistanceGoal,
)
from giskardpy.motion_statechart.tasks.grasp_bar import GraspBar
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionLimitList,
    JointPositionList,
    AvoidJointLimits,
    MirrorJointPosition,
)
from giskardpy.motion_statechart.tasks.pointing import Pointing
from giskardpy.motion_statechart.tasks.task import WEIGHT_ABOVE_CA
from giskardpy.motion_statechart.tasks.weight_scaling_goals import BaseArmWeightScaling
from giskardpy.motion_statechart.tasks.weight_scaling_goals import (
    MaxManipulability,
)
from giskardpy.motion_statechart.tasks.wiggle_insert import WiggleInsert
from giskardpy.utils.utils import get_all_classes_in_package, ImmutableDict
from giskardpy_ros.goals.realtime_goals import (
    RealTimePointing,
    CarryMyBullshit,
    FollowNavPath,
)
from giskardpy_ros.ros2 import msg_converter, rospy
from giskardpy_ros.ros2.msg_converter import kwargs_to_json
from giskardpy_ros.ros2.my_multithreaded_executor import MyMultiThreadedExecutor
from giskardpy_ros.ros2.ros2_interface import MyActionClient
from semantic_world.datastructures.prefixed_name import PrefixedName
from semantic_world.robots.abstract_robot import AbstractRobot
from semantic_world.world import World


@dataclass
class MotionStatechartNodeWrapper:
    giskard_wrapper: GiskardWrapper
    _motion_graph_nodes: Dict[str, MotionStatechartNode] = field(
        default_factory=ImmutableDict
    )
    _name_prefix: str = ""

    @property
    def robot_name(self) -> PrefixedName:
        return self.giskard_wrapper.robot_name

    @property
    def motion_graph_nodes(self) -> Dict[str, MotionStatechartNode]:
        return self._motion_graph_nodes

    def reset(self):
        self._motion_graph_nodes = ImmutableDict()

    def _add_motion_statechart_node(
        self,
        *,
        class_name: str,
        name: Optional[str] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        reset_condition: str = "",
        **kwargs,
    ) -> str:
        """
        Generic function to add a motion goal.
        :param motion_goal_class: Name of a class defined in src/giskardpy/goals
        :param name: a unique name for the goal, will use class name by default
        :param start_condition: a logical expression to define the start condition for this monitor. e.g.
                                    not 'monitor1' and ('monitor2' or 'monitor3')
        :param pause_condition: a logical expression. Goal will be on hold if it is True and active otherwise
        :param end_condition: a logical expression. Goal will become inactive when this becomes True.
        :param kwargs: kwargs for __init__ function of motion_goal_class
        """
        if name is None:
            name = f"{self._name_prefix}{len(self._motion_graph_nodes)} [{class_name}]"
        motion_goal = MotionStatechartNode()
        motion_goal.name = name
        motion_goal.class_name = class_name
        self._motion_graph_nodes[name] = motion_goal
        motion_goal.kwargs = kwargs_to_json(kwargs)

        self.update_start_condition(node_name=name, condition=start_condition)
        self.update_pause_condition(node_name=name, condition=pause_condition)
        if end_condition is None:  # everything ends themselves by default
            motion_goal.end_condition = name
            self.update_end_condition(node_name=name, condition=name)
        else:
            self.update_end_condition(node_name=name, condition=end_condition)
        self.update_reset_condition(node_name=name, condition=reset_condition)
        return name

    def get_anded_nodes(self, add_nodes_without_end_condition: bool = True) -> str:
        nodes = []
        for node in self.motion_graph_nodes.values():
            if node.class_name not in get_all_classes_in_package(
                "giskardpy.motion_statechart.monitors", CancelMotion
            ) and (add_nodes_without_end_condition or node.end_condition != ""):
                nodes.append(node.name)
        return " and ".join(nodes)

    def set_conditions(
        self,
        node_name: str,
        start_condition: str,
        pause_condition: str,
        end_condition: str,
        reset_condition: str,
    ):
        self.update_start_condition(node_name, start_condition)
        self.update_pause_condition(node_name, pause_condition)
        self.update_end_condition(node_name, end_condition)
        self.update_reset_condition(node_name, reset_condition)

    def update_start_condition(self, node_name: str, condition: str) -> None:
        self._motion_graph_nodes[node_name].start_condition = condition

    def update_reset_condition(self, node_name: str, condition: str) -> None:
        self._motion_graph_nodes[node_name].reset_condition = condition

    def update_pause_condition(self, node_name: str, condition: str) -> None:
        self._motion_graph_nodes[node_name].pause_condition = condition

    def update_end_condition(self, node_name: str, condition: str) -> None:
        self._motion_graph_nodes[node_name].end_condition = condition


@dataclass
class MotionGoalWrapper(MotionStatechartNodeWrapper):
    _name_prefix: str = "G"
    _collision_entries: Dict[Tuple[str, str, str], List[CollisionEntry]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def reset(self):
        super().reset()
        self._collision_entries = defaultdict(list)

    def add_motion_goal(
        self,
        *,
        class_name: str,
        start_condition: str = "",
        pause_condition: str = "",
        name: Optional[str] = None,
        end_condition: Optional[str] = "",
        **kwargs,
    ) -> str:
        """
        Generic function to add a motion goal.
        :param class_name: Name of a class defined in src/giskardpy/goals
        :param name: a unique name for the goal, will use class name by default
        :param start_condition: a logical expression to define the start condition for this monitor. e.g.
                                    not 'monitor1' and ('monitor2' or 'monitor3')
        :param pause_condition: a logical expression. Goal will be on hold if it is True and active otherwise
        :param end_condition: a logical expression. Goal will become inactive when this becomes True.
        :param kwargs: kwargs for __init__ function of motion_goal_class
        """
        return super()._add_motion_statechart_node(
            class_name=class_name,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_grasp_bar(
        self,
        bar_center: PointStamped,
        bar_axis: Vector3Stamped,
        bar_length: float,
        tip_link: Union[str, PrefixedName],
        tip_grasp_axis: Vector3Stamped,
        root_link: Union[str, PrefixedName],
        name: Optional[str] = None,
        reference_linear_velocity: Optional[float] = None,
        reference_angular_velocity: Optional[float] = None,
        weight: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        """
        Like a CartesianPose but with more freedom.
        tip_link is allowed to be at any point along bar_axis, that is without bar_center +/- bar_length.
        It will align tip_grasp_axis with bar_axis, but allows rotation around it.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param tip_grasp_axis: axis of tip_link that will be aligned with bar_axis
        :param bar_center: center of the bar to be grasped
        :param bar_axis: alignment of the bar to be grasped
        :param bar_length: length of the bar to be grasped
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        """
        return self.add_motion_goal(
            class_name=GraspBar.__name__,
            root_link=root_link,
            tip_link=tip_link,
            tip_grasp_axis=tip_grasp_axis,
            bar_center=bar_center,
            bar_axis=bar_axis,
            bar_length=bar_length,
            reference_linear_velocity=reference_linear_velocity,
            reference_angular_velocity=reference_angular_velocity,
            weight=weight,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_cartesian_pose(
        self,
        goal_pose: PoseStamped,
        tip_link: Union[str, PrefixedName],
        root_link: Union[str, PrefixedName],
        name: Optional[str] = None,
        reference_linear_velocity: Optional[float] = None,
        reference_angular_velocity: Optional[float] = None,
        absolute: bool = False,
        weight: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        """
        This goal will use the kinematic chain between root and tip link to move tip link to the goal pose.
        The max velocities enforce a strict limit, but require a lot of additional constraints, thus making the
        system noticeably slower.
        The reference velocities don't enforce a strict limit, but also don't require any additional constraints.
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param absolute: if False, the goal pose is reevaluated if start_condition turns True.
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight: None = use default weight
        """
        return self.add_motion_goal(
            class_name=CartesianPose.__name__,
            goal_pose=goal_pose,
            tip_link=tip_link,
            root_link=root_link,
            reference_linear_velocity=reference_linear_velocity,
            reference_angular_velocity=reference_angular_velocity,
            weight=weight,
            name=name,
            absolute=absolute,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_base_arm_weight_scaling(
        self,
        tip_link: Union[str, PrefixedName],
        root_link: Union[str, PrefixedName],
        tip_goal: PointStamped,
        arm_joints: List[str],
        base_joints: List[str],
        gain: int = 100000,
        name: Optional[str] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        """
        This goals adds weight scaling constraints with the distance between a tip_link and its goal Position as a
        scaling expression. The larger the scaling expression the more is the base movement used to achieve
        all other constraints instead of arm movements. When the expression decreases this relation changes to favor
        arm movements instead of base movements.
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param tip_goal: the goal position
        :param arm_joints: joints of the arm that should be scaled.
        :param base_joints: joints of the base that should be scaled.
        """
        return self.add_motion_goal(
            class_name=BaseArmWeightScaling.__name__,
            name=name,
            tip_link=tip_link,
            root_link=root_link,
            tip_goal=tip_goal,
            arm_joints=arm_joints,
            base_joints=base_joints,
            gain=gain,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_maximize_manipulability(
        self,
        tip_link: Union[str, PrefixedName],
        root_link: Union[str, PrefixedName],
        name: Optional[str] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        """
        This goal will use the kinematic chain between root and tip link to move tip link to the goal pose.
        The max velocities enforce a strict limit, but require a lot of additional constraints, thus making the
        system noticeably slower.
        The reference velocities don't enforce a strict limit, but also don't require any additional constraints.
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param absolute: if False, the goal pose is reevaluated if start_condition turns True.
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight: None = use default weight
        """
        return self.add_motion_goal(
            class_name=MaxManipulability.__name__,
            name=name,
            tip_link=tip_link,
            root_link=root_link,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_align_planes(
        self,
        goal_normal: Vector3Stamped,
        tip_link: Union[str, PrefixedName],
        tip_normal: Vector3Stamped,
        root_link: Union[str, PrefixedName],
        name: Optional[str] = None,
        reference_angular_velocity: Optional[float] = None,
        weight: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        """
        This goal will use the kinematic chain between tip and root to align tip_normal with goal_normal.
        :param goal_normal:
        :param tip_link: tip link of the kinematic chain
        :param tip_normal:
        :param root_link: root link of the kinematic chain
        :param reference_angular_velocity: rad/s
        :param weight:
        """
        return self.add_motion_goal(
            class_name=AlignPlanes.__name__,
            tip_link=tip_link,
            tip_normal=tip_normal,
            root_link=root_link,
            goal_normal=goal_normal,
            max_angular_velocity=reference_angular_velocity,
            weight=weight,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_joint_position(
        self,
        goal_state: Dict[str, float],
        name: Optional[str] = None,
        weight: Optional[float] = None,
        max_velocity: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        """
        Sets joint position goals for all pairs in goal_state
        :param goal_state: maps joint_name to goal position
        :param weight: None = use default weight
        :param max_velocity: will be applied to all joints
        """
        return self.add_motion_goal(
            class_name=JointPositionList.__name__,
            goal_state=goal_state,
            weight=weight,
            max_velocity=max_velocity,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_mirror_joint_position(
        self,
        mapping: Dict[str, str],
        name: Optional[str] = None,
        weight: Optional[float] = None,
        max_velocity: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        """
        Sets joint position goals for all pairs in goal_state
        :param goal_state: maps joint_name to goal position
        :param weight: None = use default weight
        :param max_velocity: will be applied to all joints
        """
        return self.add_motion_goal(
            class_name=MirrorJointPosition.__name__,
            mapping=mapping,
            weight=weight,
            max_velocity=max_velocity,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_joint_position_limit(
        self,
        lower_upper_limits: Dict[str, Tuple[float, float]],
        name: Optional[str] = None,
        weight: Optional[float] = None,
        max_velocity: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        """
        Sets joint position goals for all pairs in goal_state
        :param goal_state: maps joint_name to goal position
        :param weight: None = use default weight
        :param max_velocity: will be applied to all joints
        """
        return self.add_motion_goal(
            class_name=JointPositionLimitList.__name__,
            lower_upper_limits=lower_upper_limits,
            weight=weight,
            max_velocity=max_velocity,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_justin_torso_limit(
        self,
        name: Optional[str] = None,
        root_link: Union[str, PrefixedName] = "torso1",
        tip_link: Union[str, PrefixedName] = "torso4",
        forward_distance: float = 0.05,
        backward_distance: float = 0.14,
        weight: float = WEIGHT_ABOVE_CA,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        return self.add_motion_goal(
            class_name=JustinTorsoLimitCart.__name__,
            root_link=root_link,
            tip_link=tip_link,
            forward_distance=forward_distance,
            backward_distance=backward_distance,
            name=name,
            weight=weight,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_cartesian_position(
        self,
        goal_point: PointStamped,
        tip_link: Union[str, PrefixedName],
        root_link: Union[str, PrefixedName],
        name: Optional[str] = None,
        reference_velocity: Optional[float] = 0.2,
        weight: Optional[float] = None,
        absolute: bool = False,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        """
        Will use kinematic chain between root_link and tip_link to move tip_link to goal_point.
        :param goal_point:
        :param tip_link: tip link of the kinematic chain
        :param root_link: root link of the kinematic chain
        :param reference_velocity: m/s
        :param weight:
        :param absolute: if False, the goal pose is reevaluated if start_condition turns True.
        """
        return self.add_motion_goal(
            class_name=CartesianPosition.__name__,
            goal_point=goal_point,
            tip_link=tip_link,
            root_link=root_link,
            reference_velocity=reference_velocity,
            weight=weight,
            absolute=absolute,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_cartesian_orientation(
        self,
        goal_orientation: QuaternionStamped,
        tip_link: Union[str, PrefixedName],
        root_link: Union[str, PrefixedName],
        name: Optional[str] = None,
        reference_velocity: Optional[float] = None,
        weight: Optional[float] = None,
        absolute: bool = False,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        """
        Will use kinematic chain between root_link and tip_link to move tip_link to goal_orientation.
        :param goal_orientation:
        :param tip_link: tip link of kinematic chain
        :param root_link: root link of kinematic chain
        :param tip_group: if tip link is not unique, you can use this to tell Giskard in which group to search.
        :param root_group: if root link is not unique, you can use this to tell Giskard in which group to search.
        :param reference_velocity: rad/s, approx limit
        :param absolute: if False, the goal pose is reevaluated if start_condition turns True.
        """
        return self.add_motion_goal(
            class_name=CartesianOrientation.__name__,
            goal_orientation=goal_orientation,
            tip_link=tip_link,
            root_link=root_link,
            reference_velocity=reference_velocity,
            weight=weight,
            absolute=absolute,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_pointing(
        self,
        goal_point: PointStamped,
        tip_link: Union[str, PrefixedName],
        pointing_axis: Vector3Stamped,
        root_link: Union[str, PrefixedName],
        name: Optional[str] = None,
        max_velocity: float = 0.3,
        threshold: float = 0.01,
        weight: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        """
        Will orient pointing_axis at goal_point.
        :param tip_link: tip link of the kinematic chain.
        :param goal_point: where to point pointing_axis at.
        :param root_link: root link of the kinematic chain.
        :param pointing_axis: the axis of tip_link that will be used for pointing
        :param max_velocity: rad/s
        """
        return self.add_motion_goal(
            class_name=Pointing.__name__,
            tip_link=tip_link,
            goal_point=goal_point,
            root_link=root_link,
            pointing_axis=pointing_axis,
            max_velocity=max_velocity,
            threshold=threshold,
            weight=weight,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_wiggle_insert(
        self,
        root_link: PrefixedName,
        tip_link: PrefixedName,
        hole_point: PointStamped,
        name: Optional[str] = None,
        down_velocity: float = 0.2,
        noise_translation: float = 0.5,
        noise_angle: float = 10,
        random_walk: bool = True,
        vector_momentum_factor: float = 0.9,
        angular_momentum_factor: float = 0.9,
        center_pull_strength_angle: float = 0.1,
        center_pull_strength_vector: float = 0.25,
        weight: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
    ):
        """
        Press down while wiggling the end effector.
        :param root_link:
        :param tip_link:
        :param hole_point: Center point of the hole
        :param hole_normal: Vector perpendicular to the hole. default = z-axis of map
        :param threshold: threshold for distance to hole_point to end task
        :param noise_translation: describes how strong the translation wiggle is.
        :param noise_angle: describes how strong the angular wiggle is.
        :param random_walk: determines if random walk or random sample strategy is used
        :param vector_momentum_factor: (only when random_walk=True) Higher value increases influence of momentum,
                                                                    creating smoother but less random translation movement
        :param angular_momentum_factor: (only when random_walk=True) Higher value increases influence of momentum,
                                                                     creating smoother but less random angular movement
        :param center_pull_strength_angle: (only when random_walk=True) Forces angular movement faster back towards
                                                                        starting angle
        :param center_pull_strength_vector: (only when random_walk=True) Forces translation movement faster back towards
                                                                         hole_point
        """
        return self.add_motion_goal(
            class_name=WiggleInsert.__name__,
            root_link=root_link,
            tip_link=tip_link,
            name=name,
            down_velocity=down_velocity,
            hole_point=hole_point,
            noise_translation=noise_translation,
            noise_angle=noise_angle,
            random_walk=random_walk,
            vector_momentum_factor=vector_momentum_factor,
            angular_momentum_factor=angular_momentum_factor,
            center_pull_strength_angle=center_pull_strength_angle,
            center_pull_strength_vector=center_pull_strength_vector,
            weight=weight,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
        )

    def _add_collision_avoidance(
        self,
        collisions: List[CollisionEntry],
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
    ):
        key = (start_condition, pause_condition, end_condition)
        self._collision_entries[key].extend(collisions)

    def _add_collision_entries_as_goals(self):
        for (
            start_condition,
            pause_condition,
            end_condition,
        ), collision_entries in self._collision_entries.items():
            if (
                collision_entries[-1].type == CollisionEntry.ALLOW_COLLISION
                and collision_entries[-1].group1 == CollisionEntry.ALL
                and collision_entries[-1].group2 == CollisionEntry.ALL
            ):
                continue
            name = "collision avoidance"
            if start_condition or pause_condition or end_condition:
                name += f"{start_condition}, {pause_condition}, {end_condition}"
            self.add_motion_goal(
                class_name=CollisionAvoidance.__name__,
                name=name,
                collision_entries=collision_entries,
                start_condition=start_condition,
                pause_condition=pause_condition,
                end_condition=end_condition,
            )

    def allow_collision(
        self,
        group1: str = CollisionEntry.ALL,
        group2: str = CollisionEntry.ALL,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
    ):
        """
        Tell Giskard to allow collision between group1 and group2. Use CollisionEntry.ALL to allow collision with all
        groups.
        :param group1: name of the first group
        :param group2: name of the second group
        """
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.group1 = str(group1)
        collision_entry.group2 = str(group2)
        self._add_collision_avoidance(
            collisions=[collision_entry],
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
        )

    def avoid_collision(
        self,
        min_distance: Optional[float] = None,
        group1: str = CollisionEntry.ALL,
        group2: str = CollisionEntry.ALL,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
    ):
        """
        Tell Giskard to avoid collision between group1 and group2. Use CollisionEntry.ALL to allow collision with all
        groups.
        :param min_distance: set this to overwrite the default distances
        :param group1: name of the first group
        :param group2: name of the second group
        """
        if min_distance is None:
            min_distance = -1.0
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.distance = min_distance
        collision_entry.group1 = group1
        collision_entry.group2 = group2
        self._add_collision_avoidance(
            collisions=[collision_entry],
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
        )

    def allow_all_collisions(
        self,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
    ):
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        self._add_collision_avoidance(
            collisions=[collision_entry],
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
        )

    def avoid_all_collisions(
        self,
        min_distance: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
    ):
        """
        If you don't want to override the distance, don't call this function. Avoid all is the default, if you don't
        add any collision entries.
        :param min_distance: set this to overwrite default distances
        """
        if min_distance is None:
            min_distance = -1.0
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.distance = min_distance
        self._add_collision_avoidance(
            collisions=[collision_entry],
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
        )

    def allow_self_collision(
        self,
        robot_name: Optional[str] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
    ):
        """
        Allows the collision of the robot with itself for the next goal.
        :param robot_name: if there are multiple robots, specify which one.
        """
        if robot_name is None:
            robot_name = self.robot_name
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        if isinstance(robot_name, PrefixedName):
            collision_entry.group1 = robot_name.name
            collision_entry.group2 = robot_name.name
        else:
            collision_entry.group1 = robot_name
            collision_entry.group2 = robot_name
        self._add_collision_avoidance(
            collisions=[collision_entry],
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
        )

    def add_avoid_joint_limits(
        self,
        name: Optional[str] = None,
        percentage: float = 15.0,
        joint_list: Optional[List[str]] = None,
        weight: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
    ) -> str:
        """
        This goal will push joints away from their position limits. For example if percentage is 15 and the joint
        limits are 0-100, it will push it into the 15-85 range.
        """
        return self.add_motion_goal(
            class_name=AvoidJointLimits.__name__,
            percentage=percentage,
            weight=weight,
            joint_list=joint_list,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
        )

    def add_close_container(
        self,
        tip_link: Union[str, PrefixedName],
        environment_link: Union[str, PrefixedName],
        name: Optional[str] = None,
        goal_joint_state: Optional[float] = None,
        weight: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
    ) -> str:
        """
        Same as Open, but will use minimum value as default for goal_joint_state
        """
        return self.add_motion_goal(
            class_name=Close.__name__,
            tip_link=tip_link,
            environment_link=environment_link,
            goal_joint_state=goal_joint_state,
            weight=weight,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
        )

    def add_open_container(
        self,
        tip_link: Union[str, LinkName],
        environment_link: Union[str, LinkName],
        name: Optional[str] = None,
        goal_joint_state: Optional[float] = None,
        weight: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
    ) -> str:
        """
        Open a container in an environment.
        Only works with the environment was added as urdf.
        Assumes that a handle has already been grasped.
        Can only handle containers with 1 dof, e.g. drawers or doors.
        :param tip_link: end effector that is grasping the handle
        :param environment_link: name of the handle that was grasped
        :param goal_joint_state: goal state for the container. default is maximum joint state.
        :param weight:
        """
        return self.add_motion_goal(
            class_name=Open.__name__,
            tip_link=tip_link,
            environment_link=environment_link,
            goal_joint_state=goal_joint_state,
            weight=weight,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
        )

    def add_align_to_push_door(
        self,
        root_link: str,
        tip_link: str,
        door_object: str,
        door_handle: str,
        tip_gripper_axis: Vector3Stamped,
        weight: Optional[float] = None,
        name: Optional[str] = None,
        tip_group: Optional[str] = None,
        root_group: Optional[str] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
    ) -> str:
        """
        Aligns the tip_link with the door_object to push it open. Only works if the door object is part of the urdf.
        The door has to be open a little before aligning.
        : param root_link: root link of the kinematic chain
        : param tip_link: end effector
        : param door object: name of the object to be pushed
        : param door_height: height of the door
        : param door_handle: name of the object handle
        : param object_joint_name: name of the joint that rotates
        : param tip_gripper_axis: axis of the tip_link that will be aligned along the door rotation axis
        : param object_rotation_axis: door rotation axis w.r.t root
        """
        return self.add_motion_goal(
            class_name=AlignToPushDoor.__name__,
            root_link=root_link,
            tip_link=tip_link,
            door_handle=door_handle,
            door_object=door_object,
            tip_gripper_axis=tip_gripper_axis,
            tip_group=tip_group,
            root_group=root_group,
            weight=weight,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
        )

    def add_pre_push_door(
        self,
        root_link: str,
        tip_link: str,
        door_object: str,
        door_handle: str,
        weight: Optional[float] = None,
        name: Optional[str] = None,
        tip_group: Optional[str] = None,
        root_group: Optional[str] = None,
        reference_linear_velocity: Optional[float] = None,
        reference_angular_velocity: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
    ) -> str:
        """
        Positions the gripper in contact with the door before pushing to open.
        : param root_link: root link of the kinematic chain
        : param tip_link: end effector
        : param door object: name of the object to be pushed
        : param door_handle: name of the object handle
        : param root_V_object_rotation_axis: door rotation axis w.r.t root
        : param root_V_object_normal: door normal w.r.t root
        """
        return self.add_motion_goal(
            class_name=PrePushDoor.__name__,
            root_link=root_link,
            tip_link=tip_link,
            door_object=door_object,
            door_handle=door_handle,
            tip_group=tip_group,
            root_group=root_group,
            weight=weight,
            name=name,
            reference_linear_velocity=reference_linear_velocity,
            reference_angular_velocity=reference_angular_velocity,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
        )

    def add_diff_drive_base(
        self,
        goal_pose: PoseStamped,
        tip_link: Union[str, PrefixedName],
        root_link: Union[str, PrefixedName],
        name: Optional[str] = None,
        reference_linear_velocity: Optional[float] = None,
        reference_angular_velocity: Optional[float] = None,
        weight: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose.
        It is specifically for differential drives. Will drive towards the goal the following way:
        1. orient to goal
        2. drive to goal position in a straight line
        3. orient to goal orientation
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        """
        return self.add_motion_goal(
            class_name=DiffDriveBaseGoal.__name__,
            goal_pose=goal_pose,
            tip_link=tip_link,
            root_link=root_link,
            reference_linear_velocity=reference_linear_velocity,
            reference_angular_velocity=reference_angular_velocity,
            weight=weight,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_limit_cartesian_velocity(
        self,
        tip_link: Union[str, PrefixedName],
        root_link: Union[str, PrefixedName],
        name: Optional[str] = None,
        max_linear_velocity: float = 0.1,
        max_angular_velocity: float = 0.5,
        weight: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        """
        This goal will use put a strict limit on the Cartesian velocity. This will require a lot of constraints, thus
        slowing down the system noticeably.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param max_linear_velocity: m/s
        :param max_angular_velocity: rad/s
        """
        return self.add_motion_goal(
            class_name=CartesianVelocityLimit.__name__,
            root_link=root_link,
            tip_link=tip_link,
            weight=weight,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_real_time_pointing(
        self,
        tip_link: str,
        pointing_axis: Vector3Stamped,
        root_link: str,
        topic_name: str,
        name: Optional[str] = None,
        tip_group: Optional[str] = None,
        root_group: Optional[str] = None,
        max_velocity: float = 0.3,
        weight: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        """
        Will orient pointing_axis at goal_point.
        :param tip_link: tip link of the kinematic chain.
        :param topic_name: name of a topic of type PointStamped
        :param root_link: root link of the kinematic chain.
        :param tip_group: if tip_link is not unique, search this group for matches.
        :param root_group: if root_link is not unique, search this group for matches.
        :param pointing_axis: the axis of tip_link that will be used for pointing
        :param max_velocity: rad/s
        """
        return self.add_motion_goal(
            class_name=RealTimePointing.__name__,
            tip_link=tip_link,
            tip_group=tip_group,
            root_link=root_link,
            topic_name=topic_name,
            root_group=root_group,
            pointing_axis=pointing_axis,
            max_velocity=max_velocity,
            weight=weight,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_carry_my_luggage(
        self,
        name: Optional[str] = None,
        tracked_human_position_topic_name: str = "/robokudovanessa/human_position",
        laser_topic_name: str = "/hsrb/base_scan",
        point_cloud_laser_topic_name: Optional[str] = None,
        odom_joint_name: str = "brumbrum",
        root_link: Optional[str] = None,
        camera_link: str = "head_rgbd_sensor_link",
        distance_to_target_stop_threshold: float = 1,
        laser_scan_age_threshold: float = 2,
        laser_distance_threshold: float = 0.5,
        laser_distance_threshold_width: float = 0.8,
        laser_avoidance_angle_cutout: float = np.pi / 4,
        laser_avoidance_sideways_buffer: float = 0.04,
        base_orientation_threshold: float = np.pi / 16,
        tracked_human_position_topic_name_timeout: int = 30,
        max_rotation_velocity: float = 0.5,
        max_rotation_velocity_head: float = 1,
        max_translation_velocity: float = 0.38,
        traj_tracking_radius: float = 0.4,
        height_for_camera_target: float = 1,
        laser_frame_id: str = "base_range_sensor_link",
        target_age_threshold: float = 2,
        target_age_exception_threshold: float = 5,
        clear_path: bool = False,
        drive_back: bool = False,
        enable_laser_avoidance: bool = True,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
    ) -> str:
        """
        :param name: name of the goal
        :param tracked_human_position_topic_name: name of the topic where the tracked human is published
        :param laser_topic_name: topic name of the laser scanner
        :param point_cloud_laser_topic_name: topic name of a second laser scanner, e.g. from a point cloud to laser scanner node
        :param odom_joint_name: name of the odom joint
        :param root_link: will use global reference frame
        :param camera_link: link of the camera that will point to the tracked human
        :param distance_to_target_stop_threshold: will pause if closer than this many meter to the target
        :param laser_scan_age_threshold: giskard will complain if scans are older than this many seconds
        :param laser_distance_threshold: this and width are used to crate a stopping zone around the robot.
                                            laser distance draws a circle around the robot and width lines to the left and right.
                                            the stopping zone is the minimum of the two.
        :param laser_distance_threshold_width: see laser_distance_threshold
        :param laser_avoidance_angle_cutout: if something is in the stop zone in front of the robot in +/- this angle range
                                                giskard will pause, otherwise it will try to dodge left or right
        :param laser_avoidance_sideways_buffer: increase this if the robot is shaking too much if something is to its
                                                left and right at the same time.
        :param base_orientation_threshold: giskard will align the base of the robot to the target, this is a +/- buffer to avoid shaking
        :param tracked_human_position_topic_name_timeout: on start up, wait this long for tracking msg to arrive
        :param max_rotation_velocity: how quickly the base can change orientation
        :param max_rotation_velocity_head: how quickly the head rotates
        :param max_translation_velocity: how quickly the base drives
        :param traj_tracking_radius: how close the robots root link will try to stick to the path in meter
        :param height_for_camera_target: target tracking with head will ignore the published height, but use this instead
        :param laser_frame_id: frame_id of the laser scanner
        :param target_age_threshold: will stop looking at the target if the messages are older than this many seconds
        :param target_age_exception_threshold: if there are no messages from the tracked_human_position_topic_name
                                                            topic for this many seconds, cancel
        :param clear_path: clear the saved path. if called repeated will, giskard would just continue the old path if not cleared
        :param drive_back: follow the saved path to drive back
        :param enable_laser_avoidance:
        :param start_condition:
        :param pause_condition:
        :param end_condition:
        """
        return self.add_motion_goal(
            class_name=CarryMyBullshit.__name__,
            name=name,
            patrick_topic_name=tracked_human_position_topic_name,
            laser_topic_name=laser_topic_name,
            point_cloud_laser_topic_name=point_cloud_laser_topic_name,
            odom_joint_name=odom_joint_name,
            root_link=root_link,
            camera_link=camera_link,
            distance_to_target_stop_threshold=distance_to_target_stop_threshold,
            laser_scan_age_threshold=laser_scan_age_threshold,
            laser_distance_threshold=laser_distance_threshold,
            laser_distance_threshold_width=laser_distance_threshold_width,
            laser_avoidance_angle_cutout=laser_avoidance_angle_cutout,
            laser_avoidance_sideways_buffer=laser_avoidance_sideways_buffer,
            base_orientation_threshold=base_orientation_threshold,
            wait_for_patrick_timeout=tracked_human_position_topic_name_timeout,
            max_rotation_velocity=max_rotation_velocity,
            max_rotation_velocity_head=max_rotation_velocity_head,
            max_translation_velocity=max_translation_velocity,
            traj_tracking_radius=traj_tracking_radius,
            height_for_camera_target=height_for_camera_target,
            laser_frame_id=laser_frame_id,
            target_age_threshold=target_age_threshold,
            target_age_exception_threshold=target_age_exception_threshold,
            clear_path=clear_path,
            drive_back=drive_back,
            enable_laser_avoidance=enable_laser_avoidance,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
        )

    def add_follow_nav_path(
        self,
        path: Path,
        name: Optional[str] = None,
        laser_topics: Tuple[str] = ("/hsrb/base_scan",),
        odom_joint_name: Optional[str] = None,
        root_link: Optional[str] = None,
        camera_link: str = "head_rgbd_sensor_link",
        distance_to_target_stop_threshold: float = 1,
        laser_scan_age_threshold: float = 2,
        laser_distance_threshold: float = 0.5,
        laser_distance_threshold_width: float = 0.8,
        laser_avoidance_angle_cutout: float = np.pi / 4,
        laser_avoidance_sideways_buffer: float = 0.04,
        base_orientation_threshold: float = np.pi / 16,
        max_rotation_velocity: float = 0.5,
        max_rotation_velocity_head: float = 1,
        max_translation_velocity: float = 0.38,
        traj_tracking_radius: float = 0.4,
        height_for_camera_target: float = 1,
        laser_frame_id: str = "base_range_sensor_link",
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
    ) -> str:
        """
        Will follow the path, orienting itself and the head towards the next points in the list.
        At the end orient itself according to the final orientation in it. All other orientations will be ignored.
        :param name: name of the goal
        :param path: a nav path, make sure it's ordered correctly!
        :param odom_joint_name: name of the odom joint
        :param root_link: will use global reference frame
        :param camera_link: link of the camera that will point to the tracked human
        :param laser_scan_age_threshold: giskard will complain if scans are older than this many seconds
        :param laser_distance_threshold: this and width are used to crate a stopping zone around the robot.
                                            laser distance draws a circle around the robot and width lines to the left and right.
                                            the stopping zone is the minimum of the two.
        :param laser_distance_threshold_width: see laser_distance_threshold
        :param laser_avoidance_angle_cutout: if something is in the stop zone in front of the robot in +/- this angle range
                                                giskard will pause, otherwise it will try to dodge left or right
        :param laser_avoidance_sideways_buffer: increase this if the robot is shaking too much if something is to its
                                                left and right at the same time.
        :param base_orientation_threshold: giskard will align the base of the robot to the target, this is a +/- buffer to avoid shaking
        :param max_rotation_velocity: how quickly the base can change orientation
        :param max_rotation_velocity_head: how quickly the head rotates
        :param max_translation_velocity: how quickly the base drives
        :param traj_tracking_radius: how close the robots root link will try to stick to the path in meter
        :param height_for_camera_target: target tracking with head will ignore the published height, but use this instead
        :param laser_frame_id: frame_id of the laser scanner
        :param start_condition:
        :param pause_condition:
        :param end_condition:
        """
        return self.add_motion_goal(
            class_name=FollowNavPath.__name__,
            name=name,
            path=path,
            laser_topics=laser_topics,
            odom_joint_name=odom_joint_name,
            root_link=root_link,
            camera_link=camera_link,
            laser_scan_age_threshold=laser_scan_age_threshold,
            laser_distance_threshold=laser_distance_threshold,
            laser_distance_threshold_width=laser_distance_threshold_width,
            laser_avoidance_angle_cutout=laser_avoidance_angle_cutout,
            laser_avoidance_sideways_buffer=laser_avoidance_sideways_buffer,
            base_orientation_threshold=base_orientation_threshold,
            max_rotation_velocity=max_rotation_velocity,
            max_rotation_velocity_head=max_rotation_velocity_head,
            max_translation_velocity=max_translation_velocity,
            traj_tracking_radius=traj_tracking_radius,
            height_for_camera_target=height_for_camera_target,
            laser_frame_id=laser_frame_id,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
        )

    def set_seed_configuration(
        self,
        seed_configuration: Dict[str, float],
        name: Optional[str] = None,
        group_name: Optional[str] = None,
    ):
        """
        Only meant for use with projection. Changes the world state to seed_configuration before starting planning,
        without having to plan a motion to it like with add_joint_position
        """
        raise DeprecationWarning("please use monitors.set_seed_configuration instead")

    def set_seed_odometry(
        self,
        base_pose: PoseStamped,
        name: Optional[str] = None,
        group_name: Optional[str] = None,
    ):
        """
        Only meant for use with projection. Overwrites the odometry transform with base_pose.
        """
        raise DeprecationWarning("please use monitors.set_seed_odometry instead")

    def add_cartesian_pose_straight(
        self,
        goal_pose: PoseStamped,
        tip_link: Union[str, PrefixedName],
        root_link: Union[str, PrefixedName],
        name: Optional[str] = None,
        reference_linear_velocity: Optional[float] = None,
        reference_angular_velocity: Optional[float] = None,
        weight: Optional[float] = None,
        absolute: bool = False,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose.
        The max velocities enforce a strict limit, but require a lot of additional constraints, thus making the
        system noticeably slower.
        The reference velocities don't enforce a strict limit, but also don't require any additional constraints.
        In contrast to set_cart_goal, this tries to move the tip_link in a straight line to the goal_point.
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param absolute: if False, the goal pose is reevaluated if start_condition turns True.
        """
        return self.add_motion_goal(
            class_name=CartesianPoseStraight.__name__,
            goal_pose=goal_pose,
            tip_link=tip_link,
            root_link=root_link,
            weight=weight,
            reference_linear_velocity=reference_linear_velocity,
            reference_angular_velocity=reference_angular_velocity,
            name=name,
            absolute=absolute,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_cartesian_position_straight(
        self,
        goal_point: PointStamped,
        tip_link: Union[str, PrefixedName],
        root_link: Union[str, PrefixedName],
        name: Optional[str] = None,
        reference_velocity: float = None,
        weight: Optional[float] = None,
        absolute: bool = False,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        """
        Same as set_translation_goal, but will try to move in a straight line.
        """
        return self.add_motion_goal(
            class_name=CartesianPositionStraight.__name__,
            goal_point=goal_point,
            tip_link=tip_link,
            root_link=root_link,
            reference_velocity=reference_velocity,
            weight=weight,
            name=name,
            absolute=absolute,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_align_perpendicular(
        self,
        reference_normal: Vector3Stamped,
        tip_link: Union[str, PrefixedName],
        tip_normal: Vector3Stamped,
        root_link: Union[str, PrefixedName],
        name: Optional[str] = None,
        reference_velocity: Optional[float] = None,
        weight: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        return self.add_motion_goal(
            class_name=AlignPerpendicular.__name__,
            tip_normal=tip_normal,
            reference_normal=reference_normal,
            tip_link=tip_link,
            root_link=root_link,
            max_vel=reference_velocity,
            weight=weight,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_height(
        self,
        reference_point: PointStamped,
        tip_point: PointStamped,
        tip_link: Union[str, PrefixedName],
        root_link: Union[str, PrefixedName],
        lower_limit: float,
        upper_limit: float,
        name: Optional[str] = None,
        reference_velocity: Optional[float] = None,
        weight: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        return self.add_motion_goal(
            class_name=HeightGoal.__name__,
            tip_point=tip_point,
            reference_point=reference_point,
            tip_link=tip_link,
            root_link=root_link,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            max_vel=reference_velocity,
            weight=weight,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_distance(
        self,
        reference_point: PointStamped,
        tip_point: PointStamped,
        tip_link: Union[str, PrefixedName],
        root_link: Union[str, PrefixedName],
        lower_limit: float,
        upper_limit: float,
        name: Optional[str] = None,
        reference_velocity: Optional[float] = None,
        weight: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        return self.add_motion_goal(
            class_name=DistanceGoal.__name__,
            tip_point=tip_point,
            reference_point=reference_point,
            tip_link=tip_link,
            root_link=root_link,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            max_vel=reference_velocity,
            weight=weight,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )

    def add_angle(
        self,
        reference_vector: Vector3Stamped,
        tip_link: Union[str, PrefixedName],
        tip_vector: Vector3Stamped,
        root_link: Union[str, PrefixedName],
        lower_angle: float,
        upper_angle: float,
        name: Optional[str] = None,
        reference_velocity: Optional[float] = None,
        weight: Optional[float] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        **kwargs: goal_parameter,
    ) -> str:
        return self.add_motion_goal(
            class_name=AngleGoal.__name__,
            tip_vector=tip_vector,
            reference_vector=reference_vector,
            tip_link=tip_link,
            root_link=root_link,
            lower_angle=lower_angle,
            upper_angle=upper_angle,
            max_vel=reference_velocity,
            weight=weight,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            **kwargs,
        )


@dataclass
class MonitorWrapper(MotionStatechartNodeWrapper):
    _name_prefix: str = "M"
    max_trajectory_length_set: bool = False

    def add_monitor(
        self,
        *,
        class_name: str,
        name: Optional[str] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        reset_condition: str = "",
        **kwargs,
    ) -> str:
        """
        Generic function to add a monitor.
        :param class_name: Name of a class defined in src/giskardpy/monitors
        :param name: a unique name for the goal, will use class name by default
        :param start_condition: a logical expression to define the start condition for this monitor. e.g.
                                    not 'monitor1' and ('monitor2' or 'monitor3')
        :param pause_condition: a logical expression to define the hold condition for this monitor.
        :param end_condition: a logical expression to define the end condition for this monitor.
        :param kwargs: kwargs for __init__ function of motion_goal_class
        :return: the name of the monitor with added quotes to be used in logical expressions for conditions.
        """
        return super()._add_motion_statechart_node(
            class_name=class_name,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            reset_condition=reset_condition,
            **kwargs,
        )

    def add_local_minimum_reached(
        self,
        name: Optional[str] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        reset_condition: str = "",
    ) -> str:
        """
        True if the world is currently in a local minimum.
        """
        return self.add_monitor(
            class_name=LocalMinimumReached.__name__,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            reset_condition=reset_condition,
        )

    def add_time_above(
        self,
        threshold: float,
        name: Optional[str] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        reset_condition: str = "",
    ) -> str:
        """
        True if the length of the trajectory is above threshold
        """
        return self.add_monitor(
            class_name=TimeAbove.__name__,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            threshold=threshold,
            reset_condition=reset_condition,
        )

    def add_joint_position(
        self,
        goal_state: Dict[str, float],
        name: Optional[str] = None,
        threshold: float = 0.01,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        reset_condition: str = "",
    ) -> str:
        """
        True if all joints in goal_state are closer than threshold to their respective value.
        """
        return self.add_monitor(
            class_name=JointGoalReached.__name__,
            name=name,
            goal_state=goal_state,
            threshold=threshold,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            reset_condition=reset_condition,
        )

    def add_cartesian_pose(
        self,
        root_link: Union[str, PrefixedName],
        tip_link: Union[str, PrefixedName],
        goal_pose: PoseStamped,
        name: Optional[str] = None,
        position_threshold: float = 0.01,
        orientation_threshold: float = 0.01,
        absolute: bool = False,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        reset_condition: str = "",
    ) -> str:
        """
        True if tip_link is closer than the thresholds to goal_pose.
        """
        return self.add_monitor(
            class_name=PoseReached.__name__,
            name=name,
            root_link=root_link,
            tip_link=tip_link,
            goal_pose=goal_pose,
            absolute=absolute,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            reset_condition=reset_condition,
            position_threshold=position_threshold,
            orientation_threshold=orientation_threshold,
        )

    def add_cartesian_position(
        self,
        root_link: Union[str, PrefixedName],
        tip_link: Union[str, PrefixedName],
        goal_point: PointStamped,
        name: Optional[str] = None,
        threshold: float = 0.01,
        absolute: bool = False,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        reset_condition: str = "",
    ) -> str:
        """
        True if tip_link is closer than threshold to goal_point.
        """
        return self.add_monitor(
            class_name=PositionReached.__name__,
            name=name,
            root_link=root_link,
            tip_link=tip_link,
            goal_point=goal_point,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            reset_condition=reset_condition,
            absolute=absolute,
            threshold=threshold,
        )

    def add_distance_to_line(
        self,
        root_link: Union[str, PrefixedName],
        tip_link: Union[str, PrefixedName],
        center_point: PointStamped,
        line_axis: Vector3Stamped,
        line_length: float,
        name: Optional[str] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        reset_condition: str = "",
        threshold: float = 0.01,
    ) -> str:
        """
        True if tip_link is closer than threshold to the line defined by center_point, line_axis and line_length.
        """
        return self.add_monitor(
            class_name=DistanceToLine.__name__,
            name=name,
            center_point=center_point,
            line_axis=line_axis,
            line_length=line_length,
            root_link=root_link,
            tip_link=tip_link,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            reset_condition=reset_condition,
            threshold=threshold,
        )

    def add_cartesian_orientation(
        self,
        root_link: Union[str, PrefixedName],
        tip_link: Union[str, PrefixedName],
        goal_orientation: QuaternionStamped,
        name: Optional[str] = None,
        threshold: float = 0.01,
        absolute: bool = False,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        reset_condition: str = "",
    ) -> str:
        """
        True if tip_link is closer than threshold to goal_orientation
        """
        return self.add_monitor(
            class_name=OrientationReached.__name__,
            name=name,
            root_link=root_link,
            tip_link=tip_link,
            goal_orientation=goal_orientation,
            absolute=absolute,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            reset_condition=reset_condition,
            threshold=threshold,
        )

    def add_pointing_at(
        self,
        goal_point: PointStamped,
        tip_link: Union[str, PrefixedName],
        pointing_axis: Vector3Stamped,
        root_link: Union[str, PrefixedName],
        name: Optional[str] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        reset_condition: str = "",
        threshold: float = 0.01,
    ) -> str:
        """
        True if pointing_axis of tip_link is pointing at goal_point withing threshold.
        """
        return self.add_monitor(
            class_name=PointingAt.__name__,
            name=name,
            tip_link=tip_link,
            goal_point=goal_point,
            root_link=root_link,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            reset_condition=reset_condition,
            pointing_axis=pointing_axis,
            threshold=threshold,
        )

    def add_vectors_aligned(
        self,
        root_link: Union[str, PrefixedName],
        tip_link: Union[str, PrefixedName],
        goal_normal: Vector3Stamped,
        tip_normal: Vector3Stamped,
        name: Optional[str] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        reset_condition: str = "",
        threshold: float = 0.01,
    ) -> str:
        """
        True if tip_normal of tip_link is aligned with goal_normal within threshold.
        """
        return self.add_monitor(
            class_name=VectorsAligned.__name__,
            name=name,
            root_link=root_link,
            tip_link=tip_link,
            goal_normal=goal_normal,
            tip_normal=tip_normal,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            reset_condition=reset_condition,
            threshold=threshold,
        )

    def add_end_motion(self, start_condition: str, name: Optional[str] = None) -> str:
        """
        Ends the motion execution/planning if all start_condition are True.
        Use this to describe when your motion should end.
        """
        return self.add_monitor(
            class_name=EndMotion.__name__,
            name=name,
            start_condition=start_condition,
            pause_condition="",
            end_condition="",
            reset_condition="",
        )

    def add_cancel_motion(
        self,
        start_condition: str,
        error: Optional[Exception] = None,
        name: Optional[str] = None,
    ) -> str:
        """
        Cancels the motion if all start_condition are True and will make Giskard return the specified error code.
        Use this to describe when failure conditions.
        """
        if error is None:
            error = Exception(start_condition)
        error = msg_converter.exception_to_error_msg(error)
        return self.add_monitor(
            class_name=CancelMotion.__name__,
            name=name,
            start_condition=start_condition,
            pause_condition="",
            end_condition="",
            reset_condition="",
            exception=error,
        )

    def add_check_trajectory_length(self, length: float = 30) -> str:
        """
        A monitor that cancels the motion if the trajectory is longer than max_trajectory_length.
        """
        self.max_trajectory_length_set = True
        length = self.add_monitor(
            name="max traj length",
            class_name=CheckMaxTrajectoryLength.__name__,
            length=length,
            start_condition="",
            pause_condition="",
            end_condition="",
            reset_condition="",
        )
        return self.add_cancel_motion(
            start_condition=length,
            error=MaxTrajectoryLengthException("traj too long"),
            name="traj too long",
        )

    def add_print(self, message: str, start_condition: str, name: str) -> str:
        """
        Debugging Monitor.
        Print a message to the terminal if all start_condition are True.
        """
        return self.add_monitor(
            class_name=Print.__name__,
            name=name,
            message=message,
            start_condition=start_condition,
            pause_condition="",
            end_condition=name,
            reset_condition="",
        )

    def add_sleep(
        self, seconds: float, name: Optional[str] = None, start_condition: str = ""
    ) -> str:
        """
        Calls rospy.sleep(seconds) when start_condition are True and turns True itself afterward.
        """
        return self.add_monitor(
            class_name=Sleep.__name__,
            name=name,
            seconds=seconds,
            start_condition=start_condition,
            pause_condition="",
            end_condition=name,
            reset_condition="",
        )

    def add_set_seed_configuration(
        self,
        seed_configuration: Dict[Union[str, PrefixedName], float],
        name: Optional[str] = None,
        start_condition: str = "",
    ) -> str:
        """
        Only meant for use with projection. Changes the world state to seed_configuration before starting planning,
        without having to plan a motion to it like with add_joint_position
        """
        return self.add_monitor(
            class_name=SetSeedConfiguration.__name__,
            seed_configuration=seed_configuration,
            name=name,
            start_condition=start_condition,
            pause_condition="",
            end_condition=name,
            reset_condition="",
        )

    def add_set_seed_odometry(
        self,
        base_pose: PoseStamped,
        name: Optional[str] = None,
        group_name: Optional[str] = None,
        start_condition: str = "",
    ) -> str:
        """
        Only meant for use with projection. Overwrites the odometry transform with base_pose.
        """
        return self.add_monitor(
            class_name=SetOdometry.__name__,
            group_name=group_name,
            base_pose=base_pose,
            name=name,
            start_condition=start_condition,
            pause_condition="",
            end_condition=name,
            reset_condition="",
        )

    def add_alternator(
        self,
        name: Optional[str] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        reset_condition: str = "",
        mod: int = 2,
    ) -> str:
        """
        Testing monitor.
        True if floor(trajectory_length) % mod == 0.
        """
        if name is None:
            name = Alternator.__name__ + f" % {mod}"
        return self.add_monitor(
            class_name=Alternator.__name__,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            reset_condition=reset_condition,
            mod=mod,
        )

    def add_pulse(
        self,
        after_ticks: int,
        name: Optional[str] = None,
        true_for_ticks: int = 1,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        reset_condition: str = "",
    ) -> str:
        """
        Testing monitor.
        Like add_alternator but as a PayloadMonitor.
        """
        return self.add_monitor(
            class_name=Pulse.__name__,
            name=name,
            true_for_ticks=true_for_ticks,
            after_ticks=after_ticks,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            reset_condition=reset_condition,
        )

    def add_const_true(
        self,
        name: Optional[str] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        reset_condition: str = "",
    ) -> str:
        """
        Testing monitor.
        Like add_alternator but as a PayloadMonitor.
        """
        return self.add_monitor(
            class_name=TrueMonitor.__name__,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            reset_condition=reset_condition,
        )

    def add_const_false(
        self,
        name: Optional[str] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        reset_condition: str = "",
    ) -> str:
        """
        Testing monitor.
        Like add_alternator but as a PayloadMonitor.
        """
        return self.add_monitor(
            class_name=FalseMonitor.__name__,
            name=name,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            reset_condition=reset_condition,
        )

    def add_vectors_perpendicular(
        self,
        root_link: Union[str, PrefixedName],
        tip_link: Union[str, PrefixedName],
        reference_normal: Vector3Stamped,
        tip_normal: Vector3Stamped,
        name: Optional[str] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        reset_condition: str = "",
        threshold: float = 0.01,
    ) -> str:
        """
        True if tip_normal of tip_link is perpendicular to goal_normal within threshold.
        """
        return self.add_monitor(
            class_name=PerpendicularMonitor.__name__,
            name=name,
            root_link=root_link,
            tip_link=tip_link,
            reference_normal=reference_normal,
            tip_normal=tip_normal,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            reset_condition=reset_condition,
            threshold=threshold,
        )

    def add_angle(
        self,
        root_link: Union[str, PrefixedName],
        tip_link: Union[str, PrefixedName],
        reference_vector: Vector3Stamped,
        tip_vector: Vector3Stamped,
        lower_angle: float,
        upper_angle: float,
        name: Optional[str] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        reset_condition: str = "",
    ) -> str:
        """
        True if angle between tip_vector and reference_vector is within lower and upper angle.
        """
        return self.add_monitor(
            class_name=AngleMonitor.__name__,
            name=name,
            root_link=root_link,
            tip_link=tip_link,
            reference_vector=reference_vector,
            tip_vector=tip_vector,
            lower_angle=lower_angle,
            upper_angle=upper_angle,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            reset_condition=reset_condition,
        )

    def add_height(
        self,
        root_link: Union[str, PrefixedName],
        tip_link: Union[str, PrefixedName],
        reference_point: PointStamped,
        tip_point: PointStamped,
        lower_limit: float,
        upper_limit: float,
        name: Optional[str] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        reset_condition: str = "",
    ) -> str:
        """
        True if distance along the z-axis of root_link between tip_point and reference_point
        is within lower and upper limit.
        """
        return self.add_monitor(
            class_name=HeightMonitor.__name__,
            name=name,
            root_link=root_link,
            tip_link=tip_link,
            reference_point=reference_point,
            tip_point=tip_point,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            reset_condition=reset_condition,
        )

    def add_distance(
        self,
        root_link: Union[str, PrefixedName],
        tip_link: Union[str, PrefixedName],
        reference_point: PointStamped,
        tip_point: PointStamped,
        lower_limit: float,
        upper_limit: float,
        name: Optional[str] = None,
        start_condition: str = "",
        pause_condition: str = "",
        end_condition: Optional[str] = "",
        reset_condition: str = "",
    ) -> str:
        """
        True if distance between tip_point and reference_point on the plane (that has the z-axis of
        root_link as a normal vector) is within lower and upper limit.
        """
        return self.add_monitor(
            class_name=DistanceMonitor.__name__,
            name=name,
            root_link=root_link,
            tip_link=tip_link,
            reference_point=reference_point,
            tip_point=tip_point,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            start_condition=start_condition,
            pause_condition=pause_condition,
            end_condition=end_condition,
            reset_condition=reset_condition,
        )


@dataclass
class GiskardWrapper:
    """
    Python wrapper for the ROS interface of Giskard.
    :param giskard_node_name: node name of Giskard
    """

    node_handle: Node
    giskard_node_name: str = "giskard"
    last_feedback: Move.Feedback = None
    _goal_handle: Optional[ClientGoalHandle] = None
    _goal_result: Optional[Move_Result] = None
    _result_future: Optional[Future] = None
    last_execution_state: ExecutionState = None
    world: World = None
    _client: MyActionClient = None
    monitors: MonitorWrapper = field(init=False)
    motion_goals: MotionGoalWrapper = field(init=False)

    def __post_init__(self):
        # TODO needs view modification pr merge
        self.world = god_map.world
        # self.world = fetch_world_from_service(node_handle)
        # self.model_synchronizer = ModelSynchronizer(world=self.world, node=rospy.node)
        # self.state_synchronizer = StateSynchronizer(world=self.world, node=rospy.node)
        self.monitors = MonitorWrapper(self)
        self.motion_goals = MotionGoalWrapper(self)
        giskard_topic = f"{self.giskard_node_name}/command"
        self._client = MyActionClient(self.node_handle, Move, giskard_topic)
        self.clear_motion_goals_and_monitors()
        sleep(0.3)

    def set_conditions(
        self,
        node_name: str,
        start_condition: str,
        pause_condition: str,
        end_condition: str,
        reset_condition: str,
    ):
        self.update_start_condition(node_name, start_condition)
        self.update_pause_condition(node_name, pause_condition)
        self.update_end_condition(node_name, end_condition)
        self.update_reset_condition(node_name, reset_condition)

    def update_start_condition(self, node_name: str, condition: str) -> None:
        self.motion_statechart_nodes[node_name].start_condition = condition

    def update_reset_condition(self, node_name: str, condition: str) -> None:
        self.motion_statechart_nodes[node_name].reset_condition = condition

    def update_pause_condition(self, node_name: str, condition: str) -> None:
        self.motion_statechart_nodes[node_name].pause_condition = condition

    def update_end_condition(self, node_name: str, condition: str) -> None:
        self.motion_statechart_nodes[node_name].end_condition = condition

    def quote_node_names(self, condition: str) -> str:
        operators = {"and", "or", "not", "(", ")"}
        pattern = r"(\b(?:and|or|not)\b|\(|\))"
        tokens = re.split(pattern, condition)
        result = []
        for token in tokens:
            if token in operators:
                result.append(token)
            elif token.strip() == "":
                result.append(token)
            else:
                # Check if token is already quoted
                stripped = token.strip()
                if (stripped.startswith('"') and stripped.endswith('"')) or (
                    stripped.startswith("'") and stripped.endswith("'")
                ):
                    result.append(token)
                else:
                    # Wrap stripped token in quotes, and preserve leading/trailing spaces
                    leading_spaces = len(token) - len(token.lstrip())
                    trailing_spaces = len(token) - len(token.rstrip())
                    leading = token[:leading_spaces]
                    trailing = token[len(token.rstrip()) :]
                    result.append(f'{leading}"{stripped}"{trailing}')
        return "".join(result)

    @property
    def motion_statechart_nodes(self) -> Dict[str, MotionStatechartNode]:
        return {
            **self.motion_goals.motion_graph_nodes,
            **self.monitors.motion_graph_nodes,
        }

    @property
    def num_nodes(self) -> int:
        return len(self.motion_statechart_nodes)

    def add_default_end_motion_conditions(self) -> None:
        """
        1. Adds a local minimum reached monitor and adds it as an end_condition to all previously defined motion goals.
        2. Adds an end motion monitor, start_condition = all previously defined monitors are True.
        3. Adds a cancel motion monitor, start_condition = local minimum reached mit not all other monitors are True.
        4. Adds a max trajectory length monitor, if one wasn't added already.
        """
        local_min_reached_monitor_name = self.monitors.add_local_minimum_reached(
            "local min"
        )
        # for node in self.motion_statechart_nodes.values():
        #     if node.end_condition and node.name != local_min_reached_monitor_name:
        #         node.end_condition = f'({node.end_condition}) and {local_min_reached_monitor_name}'
        #     else:
        #         node.end_condition = local_min_reached_monitor_name
        end_motion_condition = local_min_reached_monitor_name
        monitor_part = self.monitors.get_anded_nodes(
            add_nodes_without_end_condition=False
        )
        if len(monitor_part) > 0:
            end_motion_condition += f" and {monitor_part}"
        motion_goal_part = self.motion_goals.get_anded_nodes(
            add_nodes_without_end_condition=False
        )
        if len(motion_goal_part) > 0:
            if len(end_motion_condition) > 0:
                end_motion_condition += f" and {motion_goal_part}"
            else:
                end_motion_condition = motion_goal_part
        self.monitors.add_end_motion(start_condition=end_motion_condition)
        # self.monitors.add_cancel_motion(start_condition=local_min_reached_monitor_name,
        #                                 error=LocalMinimumException(f'local minimum reached'))
        if not self.monitors.max_trajectory_length_set:
            self.monitors.add_check_trajectory_length()
        self.monitors.max_trajectory_length_set = False

    def add_end_on_local_minimum(self) -> None:
        local_min_reached_monitor_name = self.monitors.add_local_minimum_reached(
            "local min reached"
        )
        self.monitors.add_end_motion(start_condition=local_min_reached_monitor_name)

    @property
    def robot_name(self) -> PrefixedName:
        return self.world.get_views_by_type(AbstractRobot)[0].name

    def clear_motion_goals_and_monitors(self):
        """
        Removes all move commands from the current goal, collision entries are left untouched.
        """
        self.motion_goals.reset()
        self.monitors.reset()

    def execute_async(self) -> Future:
        return self._send_action_goal_async(Move.Goal.EXECUTE)

    def execute(self) -> Move.Result:
        """
        :return: result from giskard
        """
        return self._send_action_goal(Move.Goal.EXECUTE)

    def projection_async(self) -> Future:
        return self._send_action_goal_async(Move.Goal.PROJECTION)

    def projection(self) -> Move.Result:
        """
        Plans, but doesn't execute the goal. Useful, if you just want to look at the planning ghost.
        :return: result from Giskard
        """
        return self._send_action_goal(Move.Goal.PROJECTION)

    def _send_action_goal_async(self, goal_type: int) -> Future:
        goal = self._create_action_goal()
        goal.type = goal_type
        return self._client.send_goal_async(goal)

    def _send_action_goal(self, goal_type: int) -> Move_Result:
        goal = self._create_action_goal()
        goal.type = goal_type
        return self._client.send_goal(goal)

    def _create_action_goal(self) -> Move.Goal:
        if not self.motion_goals._collision_entries:
            self.motion_goals.avoid_all_collisions()
        self.motion_goals._add_collision_entries_as_goals()
        action_goal = Move.Goal()
        templated_and_tasks = self._quote_conditions(self.monitors.motion_graph_nodes)
        monitors = self._quote_conditions(self.motion_goals.motion_graph_nodes)
        action_goal.nodes = templated_and_tasks + monitors
        self.clear_motion_goals_and_monitors()
        return action_goal

    def _quote_conditions(
        self, nodes: Dict[str, MotionStatechartNode]
    ) -> List[MotionStatechartNode]:
        result = []
        for node in nodes.values():
            node.start_condition = self.quote_node_names(node.start_condition)
            node.pause_condition = self.quote_node_names(node.pause_condition)
            node.end_condition = self.quote_node_names(node.end_condition)
            node.reset_condition = self.quote_node_names(node.reset_condition)
            result.append(node)
        return result

    def cancel_goal_async(self) -> Future:
        """
        Stops the goal that was last sent to Giskard.
        """
        try:
            future = self._client._goal_handle.cancel_goal_async()
        except AttributeError as e:
            raise ExecutionException(
                "Can't cancel goals, because there is no active one"
            )
        return future

    async def get_result(self):
        self.last_execution_state = await self._client.get_result()
        return self.last_execution_state

    def get_end_motion_reason(
        self, move_result: Optional[Move_Result] = None, show_all: bool = False
    ) -> Dict[str, bool]:
        """
        Analyzes a MoveResult msg to return a list of all monitors that hindered the EndMotion Monitors from becoming active.
        Uses the last received MoveResult msg from execute() or projection() when not explicitly given.
        :param move_result: the move_result msg to analyze
        :param show_all: returns the state of all monitors when show_all==True
        :return: Dict with monitor name as key and True or False as value
        """
        if not move_result and not self.last_execution_state:
            raise Exception("No MoveResult available to analyze")
        elif not move_result:
            execution_state = self.last_execution_state
        else:
            execution_state = move_result.execution_state

        result = {}
        if show_all:
            return {
                monitor.name: state
                for monitor, state in zip(
                    execution_state.monitors, execution_state.monitor_state
                )
            }

        failedEndMotion_ids = []
        monitor: MotionStatechartNode
        for idx, monitor in enumerate(execution_state.monitors):
            if (
                monitor.class_name == "EndMotion"
                and execution_state.monitor_state[idx] != 1
            ):
                failedEndMotion_ids.append(idx)

        if len(failedEndMotion_ids) == 0:
            # the end motion was successful
            return result

        def search_for_monitor_values_in_start_condition(start_condition: str):
            res = []
            for monitor, state in zip(
                execution_state.monitors, execution_state.monitor_state
            ):
                if (
                    f"'{monitor.name}'" in start_condition
                    or f'"{monitor.name}"' in start_condition
                    and state != 1
                ):
                    res.append(monitor)
            return res

        for endMotion_idx in failedEndMotion_ids:
            start_condition = execution_state.monitors[endMotion_idx].start_condition
            false_monitors = search_for_monitor_values_in_start_condition(
                start_condition=start_condition
            )
            # repeatedly search for all inactive monitors in all start_conditions directly
            # connected to the endMotion start_condition
            for idx, false_monitor in enumerate(false_monitors):
                if false_monitors[idx].start_condition != "1.0":
                    false_monitors.extend(
                        search_for_monitor_values_in_start_condition(
                            false_monitor.start_condition
                        )
                    )

            for mon in false_monitors:
                result[mon.name] = False

        return result


@dataclass
class GiskardWrapperNode(GiskardWrapper):
    is_spinning: bool = False
    node_name: str = "giskard_client"
    giskard_node_name: str = "giskard"
    avoid_name_conflict: bool = True
    context: Optional[Context] = field(kw_only=True, default=None)
    cli_args: Optional[List[str]] = field(kw_only=True, default=None)
    namespace: Optional[str] = field(kw_only=True, default=None)
    use_global_arguments: bool = field(kw_only=True, default=True)
    enable_rosout: bool = field(kw_only=True, default=True)
    start_parameter_services: bool = field(kw_only=True, default=True)
    parameter_overrides: Optional[List[Parameter]] = field(kw_only=True, default=None)
    allow_undeclared_parameters: bool = field(kw_only=True, default=False)
    automatically_declare_parameters_from_overrides: bool = field(
        kw_only=True, default=False
    )
    enable_logger_service: bool = field(kw_only=True, default=False)
    node_handle: Node = field(init=False)

    def __post_init__(self):
        self.node_handle = Node(
            self.node_name,
            context=self.context,
            cli_args=self.cli_args,
            namespace=self.namespace,
            use_global_arguments=self.use_global_arguments,
            enable_rosout=self.enable_rosout,
            start_parameter_services=self.start_parameter_services,
            parameter_overrides=self.parameter_overrides,
            allow_undeclared_parameters=self.allow_undeclared_parameters,
            automatically_declare_parameters_from_overrides=self.automatically_declare_parameters_from_overrides,
        )
        rospy.executor.add_node(self.node_handle)
        self.is_spinning = False
        super().__post_init__()

    def __spin(self):
        self.my_executor = MyMultiThreadedExecutor(
            thread_name_prefix="python interface"
        )
        self.my_executor.add_node(self.node_handle)
        self.is_spinning = True
        while rclpy.ok():
            self.my_executor.spin_once(timeout_sec=1)
        self.is_spinning = False

    def spin_in_background(self):
        self.spinner = Thread(
            target=self.__spin, daemon=False, name="background giskard wrapper spinner"
        )
        self.spinner.start()
