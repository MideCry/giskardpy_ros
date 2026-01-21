from giskardpy.model.collision_world_syncer import CollisionCheckerLib

from giskardpy_ros.configs.iai_robots.dualarm import (
    WorldWithDualarmConfig,
    DualarmStandAloneRobotInterfaceConfig,
)
from giskardpy_ros.ros2 import rospy
from rclpy import Parameter

from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy_ros.configs.behavior_tree_config import StandAloneBTConfig
from giskardpy_ros.configs.giskard import Giskard


def main():
    rospy.init_node("giskard")
    rospy.node.declare_parameters(
        namespace="", parameters=[("robot_description", Parameter.Type.STRING)]
    )
    robot_description = rospy.node.get_parameter_or("robot_description").value

    giskard = Giskard(
        world_config=WorldWithDualarmConfig(urdf=robot_description),
        robot_interface_config=DualarmStandAloneRobotInterfaceConfig(),
        behavior_tree_config=StandAloneBTConfig(publish_tf=True, debug_mode=True),
        qp_controller_config=QPControllerConfig(target_frequency=33),
        collision_checker_id=CollisionCheckerLib.bpb,
    )
    giskard.live()


if __name__ == "__main__":
    main()
