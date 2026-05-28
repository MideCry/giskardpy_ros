import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import (
    Command,
    FindExecutable,
)

from launch_ros.actions import Node


def generate_launch_description():
    daisy_xacro_file = os.path.join(
        get_package_share_directory("iai_daisy_description"),
        "robots",
        "daisy.urdf.xacro",
    )
    robot_description = Command([FindExecutable(name="xacro"), " ", daisy_xacro_file])

    return LaunchDescription(
        [
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_transform_publisher",
                output="screen",
                arguments=["0", "0", "0", "0", "0", "0", "map", "world"],
            ),
            Node(
                package="giskardpy_ros",
                executable="daisy_velocity",
                name="giskard",
                parameters=[{"robot_description": robot_description}],
                output="screen",
            ),
            Node(
                package="giskardpy_ros",
                executable="interactive_marker",
                name="giskard_interactive_marker",
                parameters=[
                    {
                        "root_links": ["map", "map"],
                        "tip_links": [
                            "left_gripper_tool_frame",
                            "right_gripper_tool_frame",
                        ],
                    }
                ],
                output="screen",
            ),
        ]
    )
