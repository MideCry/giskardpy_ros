from dataclasses import dataclass
from typing import List

from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart.graph_node import DebugExpression
from giskardpy.motion_statechart.motion_statechart import MotionStatechart


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
                    case cas.HomogeneousTransformationMatrix():
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
