#!/usr/bin/env python
from rclpy import Parameter

from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy_ros.configs.behavior_tree_config import StandAloneBTConfig
from giskardpy_ros.configs.giskard import Giskard
from giskardpy_ros.configs.iai_robots.hsr import (
    WorldWithHSRConfig,
    HSRStandaloneInterface,
)
from giskardpy_ros.ros2 import rospy


def main():
    rospy.init_node("giskard")
    rospy.node.declare_parameters(
        namespace="", parameters=[("robot_description", Parameter.Type.STRING)]
    )
    robot_description = rospy.node.get_parameter_or("robot_description").value
    drive_joint_name = "brumbrum"
    giskard = Giskard(
        world_config=WorldWithHSRConfig(urdf=robot_description),
        robot_interface_config=HSRStandaloneInterface(),
        behavior_tree_config=StandAloneBTConfig(
            publish_tf=True, publish_js=False, debug_mode=True
        ),
        qp_controller_config=QPControllerConfig(mpc_dt=0.05, control_dt=None),
    )
    giskard.live()


if __name__ == "__main__":
    main()
