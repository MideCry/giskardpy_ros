from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from geometry_msgs.msg import Point, Vector3
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker

import giskardpy_ros.ros2.msg_converter as msg_converter
import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.model.collision_world_syncer import Collisions
from giskardpy.model.trajectory import Trajectory
from giskardpy.motion_statechart.graph_node import DebugExpression
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.utils.decorators import clear_memo, memoize
from giskardpy_ros.ros2 import rospy
from giskardpy_ros.ros2.visualization_mode import VisualizationMode
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class DebugMarkerVisualizer:
    node_handle: Node
    topic_suffix: str = "debug_markers"

    def __post_init__(self):
        self.publisher = self.node_handle.create_publisher(
            MarkerArray, f"{self.node_handle.get_name()}/{self.topic_suffix}", 10
        )

    def create_markers(self, motion_statechart: MotionStatechart) -> MarkerArray:
        markers = MarkerArray()
        for node in motion_statechart.nodes:
            for debug_expression in node._debug_expressions:
                match debug_expression.expression:
                    case cas.TransformationMatrix():
                        new_markers = self.transformation_matrix_to_marker(
                            debug_expression
                        )
                        pass
                    case cas.RotationMatrix():
                        new_markers = self.rotation_matrix_to_marker(debug_expression)
                    case cas.Point3():
                        new_markers = self.point_to_marker(debug_expression)
                    case cas.Vector3():
                        new_markers = self.vector_to_marker(debug_expression)
                    case cas.Quaternion():
                        pass
                    case _:
                        raise ValueError(f"Unknown debug expression {debug_expression}")
                markers.markers.extend(new_markers)
        return markers

    def vector_to_marker(
        self, debug_expression: DebugExpression, width: float = 0.05
    ) -> List[Marker]:
        m = Marker()
        m.action = Marker.ADD
        m.ns = f"debug/{debug_expression.name}"
        m.id = 0
        m.header.frame_id = str(debug_expression.expression.reference_frame.name.name)
        m.pose.orientation.w = 1.0
        vector = debug_expression.expression.evaluate()
        m.points.append(Point(x=0.0, y=0.0, z=0.0))
        m.points.append(Point(x=vector[0], y=vector[1], z=vector[2]))
        m.type = Marker.ARROW
        m.color = ColorRGBA(
            r=debug_expression.color.R,
            g=debug_expression.color.G,
            b=debug_expression.color.B,
            a=debug_expression.color.A,
        )
        m.scale.x = width / 2.0
        m.scale.y = width
        m.scale.z = 0.0
        return [m]

    def point_to_marker(
        self, debug_expression: DebugExpression, width: float = 0.05
    ) -> List[Marker]:
        m = Marker()
        m.header.frame_id = str(debug_expression.expression.reference_frame.name.name)
        m.ns = f"debug/{debug_expression.name}"
        point = debug_expression.expression.evaluate()
        m.pose.position.x = point[0]
        m.pose.position.y = point[1]
        m.pose.position.z = point[2]
        m.pose.orientation.w = 1.0
        m.type = Marker.SPHERE
        m.color = ColorRGBA(
            r=debug_expression.color.R,
            g=debug_expression.color.G,
            b=debug_expression.color.B,
            a=debug_expression.color.A,
        )
        m.scale.x = width
        m.scale.y = width
        m.scale.z = width
        return [m]

    def rotation_matrix_to_marker(
        self, debug_expression: DebugExpression, width: float = 0.05, scale: float = 0.2
    ) -> List[Marker]:
        colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),  # Red (X)
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),  # Green (Y)
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),  # Blue (Z)
        ]
        vectors = [
            debug_expression.expression.x_vector().evaluate() * scale,
            debug_expression.expression.y_vector().evaluate() * scale,
            debug_expression.expression.z_vector().evaluate() * scale,
        ]
        ms = []
        for i, axis in enumerate(vectors):
            m = Marker()
            m.header.frame_id = str(
                debug_expression.expression.reference_frame.name.name
            )
            m.pose.orientation.w = 1.0
            m.ns = f"debug/{debug_expression.name}"
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.points = [
                Point(),
                Point(x=axis[0], y=axis[1], z=axis[2]),
            ]
            m.scale.x = width / 2
            m.scale.y = width
            m.scale.z = 0.0

            m.color = colors[i]

            ms.append(m)
        return ms

    def transformation_matrix_to_marker(
        self, debug_expression: DebugExpression, width: float = 0.05, scale: float = 0.2
    ) -> List[Marker]:
        root_P_child = debug_expression.expression.to_position().evaluate()
        child_V_x = (
            debug_expression.expression.to_rotation_matrix().x_vector().evaluate()
            * scale
        )
        child_V_y = (
            debug_expression.expression.to_rotation_matrix().y_vector().evaluate()
            * scale
        )
        child_V_z = (
            debug_expression.expression.to_rotation_matrix().z_vector().evaluate()
            * scale
        )
        colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),  # Red (X)
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),  # Green (Y)
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),  # Blue (Z)
        ]
        vectors = [child_V_x, child_V_y, child_V_z]
        ms = []
        for i, axis in enumerate(vectors):
            m = Marker()
            m.header.frame_id = str(
                debug_expression.expression.reference_frame.name.name
            )
            m.pose.orientation.w = 1.0
            m.ns = f"debug/{debug_expression.name}"
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.points = [
                Point(x=root_P_child[0], y=root_P_child[1], z=root_P_child[2]),
                Point(
                    x=root_P_child[0] + axis[0],
                    y=root_P_child[1] + axis[1],
                    z=root_P_child[2] + axis[2],
                ),
            ]
            m.scale.x = width / 2
            m.scale.y = width
            m.scale.z = 0.0

            m.color = colors[i]

            ms.append(m)
        return ms

    def publish_markers(self, motion_statechart: MotionStatechart):
        markers = self.create_markers(motion_statechart)
        self.publisher.publish(markers)


class ROSMsgVisualization:
    red = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
    yellow = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
    green = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
    colors = [
        red,  # red
        ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),  # blue
        yellow,  # yellow
        ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0),  # violet
        ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),  # cyan
        green,  # green
        ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),  # white
        ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0),  # black
    ]
    mode: VisualizationMode
    frame_locked: bool
    world_version: int

    def __init__(
        self,
        tf_frame: Optional[str] = None,
        visualization_topic: str = "~visualization_marker_array",
        scale_scale: float = 1.0,
        mode: VisualizationMode = VisualizationMode.CollisionsDecomposed,
    ):
        self.mode = mode
        self.scale_scale = scale_scale
        self.frame_locked = self.mode in [
            VisualizationMode.VisualsFrameLocked,
            VisualizationMode.CollisionsFrameLocked,
            VisualizationMode.CollisionsDecomposedFrameLocked,
        ]
        # qos_profile = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = rospy.node.create_publisher(
            MarkerArray, f"{rospy.node.get_name()}/visualization_marker_array", 10
        )
        self.publisher_aux = rospy.node.create_publisher(
            MarkerArray, f"{rospy.node.get_name()}/visualization_marker_array/aux", 10
        )
        self.marker_ids = {}
        if tf_frame is None:
            self.tf_root = str(GiskardBlackboard().executor.world.root.name.name)
        else:
            self.tf_root = tf_frame
        GiskardBlackboard().ros_visualizer = self
        self.world_version = -1

    @memoize
    def link_to_marker(self, link: Body) -> List[Marker]:
        ms = msg_converter.link_to_visualization_marker(
            data=link, mode=self.mode
        ).markers
        for m in ms:
            m.scale.x *= self.scale_scale
            m.scale.y *= self.scale_scale
            m.scale.z *= self.scale_scale
        return ms

    def clear_marker_cache(self) -> None:
        clear_memo(self.link_to_marker)

    def has_world_changed(self) -> bool:
        if (
            self.world_version
            != GiskardBlackboard()
            .giskard.executor.world.get_world_model_manager()
            .version
        ):
            self.world_version = (
                GiskardBlackboard()
                .giskard.executor.world.get_world_model_manager()
                .version
            )
            return True
        return False

    def create_world_markers(
        self, name_space: str = "world", marker_id_offset: int = 0
    ) -> List[Marker]:
        markers = []
        time_stamp = rospy.node.get_clock().now().to_msg()
        if self.mode in [
            VisualizationMode.Visuals,
            VisualizationMode.VisualsFrameLocked,
        ]:
            bodies = GiskardBlackboard().executor.world.bodies
        else:
            bodies = GiskardBlackboard().executor.world.link_names_with_collisions
        for i, body in enumerate(bodies):
            collision_markers = self.link_to_marker(body)
            for j, marker in enumerate(collision_markers):
                if self.frame_locked:
                    marker.header.frame_id = body.name.name
                else:
                    marker.header.frame_id = self.tf_root
                marker.action = Marker.ADD
                link_id_key = f"{body.name}_{j}"
                if link_id_key not in self.marker_ids:
                    self.marker_ids[link_id_key] = len(self.marker_ids)
                marker.id = self.marker_ids[link_id_key] + marker_id_offset
                marker.ns = name_space
                marker.header.stamp = time_stamp
                if self.frame_locked:
                    marker.frame_locked = True
                else:
                    marker.pose = GiskardBlackboard().executor.collision_scene.collision_detector.get_map_T_geometry(
                        body.name, j
                    )
                markers.append(marker)
        return markers

    def create_collision_markers(self, name_space: str = "collisions") -> List[Marker]:
        try:
            collisions: Collisions = (
                GiskardBlackboard().executor.collision_scene.closest_points
            )
        except AttributeError as e:
            # no collisions
            return []
        if len(collisions.all_collisions) == 0:
            return []
        m = Marker()
        m.header.frame_id = self.tf_root
        m.action = Marker.ADD
        m.type = Marker.LINE_LIST
        m.id = 1337
        m.ns = name_space
        m.scale = Vector3(x=0.003, y=0.0, z=0.0)
        m.pose.orientation.w = 1.0
        if len(collisions.all_collisions) > 0:
            for collision in collisions.all_collisions:
                red_threshold = max(
                    collision.body_a.get_collision_config().violated_distance or 0.0,
                    collision.body_b.get_collision_config().violated_distance or 0.0,
                )
                yellow_threshold = max(
                    collision.body_a.get_collision_config().buffer_zone_distance or 0.0,
                    collision.body_b.get_collision_config().buffer_zone_distance or 0.0,
                )
                contact_distance = collision.contact_distance
                if collision.map_P_pa is None:
                    map_T_a = GiskardBlackboard().executor.world.compute_forward_kinematics_np(
                        GiskardBlackboard().executor.world.root,
                        GiskardBlackboard().executor.world.get_kinematic_structure_entity_by_name(
                            collision.original_body_a
                        ),
                    )
                    map_P_pa = np.dot(map_T_a, collision.a_P_pa)
                else:
                    map_P_pa = collision.map_P_pa

                if collision.map_P_pb is None:
                    map_T_b = GiskardBlackboard().executor.world.compute_forward_kinematics_np(
                        GiskardBlackboard().executor.world.root,
                        GiskardBlackboard().executor.world.get_kinematic_structure_entity_by_name(
                            collision.original_body_b
                        ),
                    )
                    map_P_pb = np.dot(map_T_b, collision.b_P_pb)
                else:
                    map_P_pb = collision.map_P_pb
                m.points.append(Point(x=map_P_pa[0], y=map_P_pa[1], z=map_P_pa[2]))
                m.points.append(Point(x=map_P_pb[0], y=map_P_pb[1], z=map_P_pb[2]))
                m.colors.append(self.red)
                m.colors.append(self.green)
                if contact_distance < yellow_threshold:
                    # m.colors[-2] = self.yellow
                    m.colors[-1] = self.yellow
                if contact_distance < red_threshold:
                    # m.colors[-2] = self.red
                    m.colors[-1] = self.red
        else:
            return []
        return [m]

    def publish_markers(
        self,
        world_ns: str = "world",
        collision_ns: str = "collisions",
        force: bool = False,
    ) -> None:
        if not self.mode == VisualizationMode.Nothing:
            marker_array = MarkerArray()
            if force or (
                not self.frame_locked or self.frame_locked and self.has_world_changed()
            ):
                self.clear_marker(world_ns)
                marker_array.markers.extend(
                    self.create_world_markers(name_space=world_ns)
                )
            marker_array.markers.extend(
                self.create_collision_markers(name_space=collision_ns)
            )
            if len(marker_array.markers) > 0:
                self.publisher.publish(marker_array)

    def publish_trajectory_markers(
        self,
        trajectory: Trajectory,
        every_x: int = 10,
        start_alpha: float = 0.5,
        stop_alpha: float = 1.0,
        namespace: str = "trajectory",
    ) -> None:
        self.clear_marker(namespace)
        marker_array = MarkerArray()

        def compute_alpha(i):
            if i < 0 or i >= len(trajectory):
                raise ValueError(f"Index {i} is out of range {len(trajectory)}")
            return start_alpha + i * (stop_alpha - start_alpha) / (len(trajectory) - 1)

        with GiskardBlackboard().executor.world.reset_joint_state_context():
            for point_id, joint_state in trajectory.items():
                if point_id % every_x == 0 or point_id == len(trajectory) - 1:
                    GiskardBlackboard().executor.world.state = joint_state
                    GiskardBlackboard().executor.world.notify_state_change()
                    if self.mode not in [
                        VisualizationMode.Visuals,
                        VisualizationMode.VisualsFrameLocked,
                    ]:
                        GiskardBlackboard().executor.collision_scene.sync()
                    markers = self.create_world_markers(
                        name_space=namespace, marker_id_offset=len(marker_array.markers)
                    )
                    for m in markers:
                        m.color.a = compute_alpha(point_id)
                    marker_array.markers.extend(deepcopy(markers))
        self.publisher.publish(marker_array)

    def clear_marker(self, ns: str):
        msg = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        marker.ns = ns
        msg.markers.append(marker)
        self.publisher.publish(msg)
        # self.marker_ids = {}

    def pub_box_marker(
        self, name: str, frame_id: str, xyz: List[float], color: ColorRGBA
    ) -> None:
        m = Marker()
        m.scale.x = xyz[0]
        m.scale.y = xyz[1]
        m.scale.z = xyz[2]
        m.frame_locked = True
        m.header.frame_id = frame_id
        m.ns = name
        m.id = 3213
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.color = color
        m.pose.orientation.w = 1.0
        marker_array = MarkerArray()
        marker_array.markers.append(m)
        self.publisher_aux.publish(marker_array)
