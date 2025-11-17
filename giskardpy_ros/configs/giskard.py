from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import List

import rclpy

from giskardpy.data_types.exceptions import SetupException
from giskardpy.executor import Executor
from giskardpy.middleware import get_middleware
from giskardpy.model.collision_world_syncer import (
    CollisionCheckerLib,
)
from giskardpy.model.world_config import WorldConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy_ros.configs.behavior_tree_config import BehaviorTreeConfig
from giskardpy_ros.configs.robot_interface_config import RobotInterfaceConfig
from giskardpy_ros.ros2 import rospy
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard
from semantic_digital_twin.adapters.ros.world_fetcher import FetchWorldServer
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    ModelSynchronizer,
    StateSynchronizer,
)
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.world_description.connections import ActiveConnection


@dataclass
class Giskard:
    """
    The main Class of Giskard.
    Instantiate it with appropriate configs for you setup and then call giskard.live()
    :param world_config: A world configuration. Use a predefined one or implement your own WorldConfig class.
    :param robot_interface_config: How Giskard talk to the robot. You probably have to implement your own RobotInterfaceConfig.
    :param collision_avoidance_config: default is no collision avoidance or implement your own collision_avoidance_config.
    :param behavior_tree_config: default is open loop mode
    :param qp_controller_config: default is good for almost all cases
    :param additional_goal_package_paths: specify paths that Giskard needs to import to find your custom Goals.
                                          Giskard will run 'from <additional path> import *' for each additional
                                          path in the list.
    :param additional_monitor_package_paths: specify paths that Giskard needs to import to find your custom Monitors.
                                          Giskard will run 'from <additional path> import *' for each additional
                                          path in the list.
    """

    world_config: WorldConfig
    behavior_tree_config: BehaviorTreeConfig
    robot_interface_config: RobotInterfaceConfig
    collision_checker_id: CollisionCheckerLib = CollisionCheckerLib.bpb
    qp_controller_config: QPControllerConfig = field(default_factory=QPControllerConfig)
    executor: Executor = field(init=False)
    model_synchronizer: ModelSynchronizer = field(init=False)
    state_synchronizer: StateSynchronizer = field(init=False)
    world_fetcher: FetchWorldServer = field(init=False)
    tmp_folder: str = field(
        default_factory=lambda: get_middleware().resolve_iri(
            "package://giskardpy_ros/tmp/"
        )
    )

    def __post_init__(self):
        GiskardBlackboard().giskard = self

    def setup(self):
        """
        Initialize the behavior tree and world. You usually don't need to call this.
        """
        with self.world_config.world.modify_world():
            self.world_config.setup_world()
            self.world_config.world.__class__.root.fget.cache_clear()
            self.executor = Executor(
                world=self.world_config.world,
                controller_config=self.qp_controller_config,
                collision_checker=self.collision_checker_id,
                tmp_folder=self.tmp_folder,
            )

            self.behavior_tree_config.setup()

            self.robot_interface_config.setup()
            self.world_config.setup_collision_config()

        if self.executor.collision_scene.is_collision_checking_enabled():
            self.executor.collision_scene.sync()

        self.sanity_check()
        self.setup_world_model_ros_interface()
        GiskardBlackboard().tree.setup(rospy.node)

    def setup_world_model_ros_interface(self):
        self.model_synchronizer = ModelSynchronizer(
            world=self.world_config.world, node=rospy.node
        )
        self.model_synchronizer.pause()
        self.state_synchronizer = StateSynchronizer(
            world=self.world_config.world, node=rospy.node
        )
        self.world_fetcher = FetchWorldServer(
            node=rospy.node, world=self.world_config.world
        )

    def sanity_check(self):
        self._controlled_joints_sanity_check()

    @property
    def robot(self) -> AbstractRobot:
        return self.robots[0]

    @property
    def robots(self) -> List[AbstractRobot]:
        return self.world_config.world.get_semantic_annotations_by_type(AbstractRobot)

    def _controlled_joints_sanity_check(self):
        world = self.world_config.world
        movable_joints = world.get_connections_by_type(ActiveConnection)
        controlled_joints = self.robot.controlled_connections
        non_controlled_joints = set(movable_joints).difference(set(controlled_joints))
        if len(controlled_joints) == 0 and len(world.connections) > 0:
            raise SetupException("No joints are flagged as controlled.")
        if len(non_controlled_joints) > 0:
            get_middleware().loginfo(
                f"The following joints are non-fixed according to the urdf, "
                f"but not flagged as controlled: {[c.name for c in non_controlled_joints]}."
            )

    def live(self):
        """
        Start Giskard.
        """
        try:
            self.setup()
            GiskardBlackboard().tree.live()
        except Exception as e:
            traceback.print_exc()
            rclpy.shutdown()
