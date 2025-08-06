from giskardpy_ros.ros2 import rospy
from rclpy import Parameter
from rclpy.exceptions import ParameterUninitializedException

from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy_ros.configs.behavior_tree_config import ClosedLoopBTConfig
from giskardpy_ros.configs.giskard import Giskard
from giskardpy_ros.configs.iai_robots.tracy import TracyVelocityInterface, TracyWorldConfig
from giskardpy_ros.ros2.visualization_mode import VisualizationMode

# left_wrist_3_joint -1.54  -2.28

# right_wrist_3_joint -4.21   -2.5



def main():
    rospy.init_node('giskard')
    try:
        rospy.node.declare_parameters(namespace='',
                                      parameters=[('robot_description', Parameter.Type.STRING)])
        robot_description = rospy.node.get_parameter_or('robot_description').value
    except ParameterUninitializedException as e:
        robot_description = None
    giskard = Giskard(world_config=TracyWorldConfig(robot_description=robot_description),
                      collision_avoidance_config=LoadSelfCollisionMatrixConfig(
                          '/home/tracy/workspace/ros/src/giskardpy_ros/self_collision_matrices/iai/tracy.srdf'),
                      robot_interface_config=TracyVelocityInterface(),
                      behavior_tree_config=ClosedLoopBTConfig(visualization_mode=VisualizationMode.VisualsFrameLocked),
                      qp_controller_config=QPControllerConfig(mpc_dt=0.0125,
                                                              control_dt=0.0125,
                                                              prediction_horizon=30))
    giskard.live()

    if __name__ == '__main__':
        main()
