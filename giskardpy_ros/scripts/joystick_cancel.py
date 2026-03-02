from giskardpy.middleware.ros2 import rospy
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy_ros.python_interface.python_interface import GiskardWrapper
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

rospy.init_node('irgendwas')

giskard = GiskardWrapper(node_handle=rospy.node)

msc = MotionStatechart()

tip = giskard.world.get_kinematic_structure_entity_by_name('hand_gripper_tool_frame')
root = giskard.world.root

goal_pose = HomogeneousTransformationMatrix.from_xyz_rpy(x=10,
                                              pitch=0.1,
                                              reference_frame=tip)

cart_goal = CartesianPose(root_link=root,
                          tip_link=tip,
                          goal_pose=goal_pose,
                          reference_linear_velocity=0.1)

msc.add_node(cart_goal)
end = EndMotion.when_true(cart_goal)
msc.add_node(end)


giskard.execute(msc)