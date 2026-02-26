#!/usr/bin/env python3
import rclpy
from action_msgs.srv import CancelGoal
from sensor_msgs.msg import Joy
import numpy as np

from giskardpy_ros.ros2 import rospy
from giskardpy_ros.ros2.ros2_interface import wait_for_message


class JoystickEStop:
    cancel_msg = "Canceling all Giskard goals."

    def __init__(self, num_buttons: int, button_ids: list, giskard_node_name: str = "giskard"):
        self.cancel_client = rospy.node.create_client(
            CancelGoal, f"{giskard_node_name}/command/_action/cancel_goal"
        )
        self.button_filter = np.zeros(num_buttons, dtype=bool)
        self.button_filter[button_ids] = True
        self.joy_sub = rospy.node.create_subscription(Joy, "/joy", self.joy_cb, 10)

    def joy_cb(self, joy_msg: Joy):
        buttons = np.array(joy_msg.buttons)
        filtered_buttons = buttons[self.button_filter]
        if np.any(filtered_buttons):
            rospy.node.get_logger().warning(
                f"joystick buttons {np.argwhere(filtered_buttons).tolist()} pressed"
            )
            rospy.node.get_logger().warning(self.cancel_msg)
            if self.cancel_client.service_is_ready():
                # An empty CancelGoal request with all-zero goal_id cancels all goals.
                self.cancel_client.call_async(CancelGoal.Request())
            else:
                rospy.node.get_logger().warning("Cancel service not ready, cannot cancel goals.")


def main(args=None):
    rospy.init_node("giskard_e_stop")

    _, joy_msg = wait_for_message(Joy, rospy.node, "/joy")
    num_buttons = len(joy_msg.buttons)

    rospy.node.declare_parameter("button_ids", list(range(num_buttons)))
    button_ids = rospy.node.get_parameter("button_ids").value

    e_stop = JoystickEStop(num_buttons, button_ids)
    rospy.node.get_logger().info("giskard joystick e stop is running")

    rospy.spinner_thread.join()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

