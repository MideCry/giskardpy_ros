from giskardpy.model.collision_world_syncer import CollisionCheckerLib

from configs.iai_robots.stretch import WorldWithStretchConfig, StretchVelocityInterface
from giskardpy_ros.configs.iai_robots.tiago import TiagoVelocityInterface, WorldWithTiagoConfig
from giskardpy_ros.ros2 import rospy
from rclpy import Parameter
from rclpy.exceptions import ParameterUninitializedException

from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy_ros.configs.behavior_tree_config import ClosedLoopBTConfig
from giskardpy_ros.configs.giskard import Giskard
from giskardpy_ros.ros2.visualization_mode import VisualizationMode
from giskardpy_ros.utils.utils import load_xacro


def main():
    rospy.init_node("giskard")
    # try:
    #     rospy.node.declare_parameters(
    #         namespace="", parameters=[("robot_description", Parameter.Type.STRING)]
    #     )
    #     robot_description = rospy.node.get_parameter_or("robot_description").value
    # except ParameterUninitializedException as e:
    robot_description = load_xacro(
        "package://stretch_urdf/stretch_urdf/RE2V0/stretch_description_RE2V0_tool_stretch_dex_wrist.urdf"
    )
    giskard = Giskard(
        world_config=WorldWithStretchConfig(urdf=robot_description),
        collision_checker_id=CollisionCheckerLib.none,
        robot_interface_config=StretchVelocityInterface(),
        behavior_tree_config=ClosedLoopBTConfig(
            visualization_mode=VisualizationMode.VisualsFrameLocked
        ),
        qp_controller_config=QPControllerConfig(
            target_frequency=80, prediction_horizon=15
        ),
    )
    giskard.live()


if __name__ == "__main__":
    main()