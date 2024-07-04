from rclpy.node import Node

import giskardpy
from giskardpy_ros.ros2.interface import ROS2Wrapper

giskardpy.middleware.middleware = ROS2Wrapper()
ros_node: Node = giskardpy.middleware.middleware.node
