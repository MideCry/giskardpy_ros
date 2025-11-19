import builtins
import json
import threading
from dataclasses import fields
from json import JSONDecodeError
from time import sleep
from typing import Optional, Union, List, Dict, Any, Type

import geometry_msgs.msg as geometry_msgs
import giskard_msgs.msg as giskard_msgs
import numpy as np
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import tf2_msgs.msg as tf2_msgs
import trajectory_msgs.msg as trajectory_msgs
import visualization_msgs.msg as visualization_msgs
from geometry_msgs.msg import TransformStamped
from random_events.utils import SubclassJSONSerializer
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy_message_converter.message_converter import (
    convert_dictionary_to_ros_message as original_convert_dictionary_to_ros_message,
    convert_ros_message_to_dictionary as original_convert_ros_message_to_dictionary,
)
from typing_extensions import get_origin, get_args

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import (
    GiskardException,
    CorruptShapeException,
    UnknownGoalException,
)
from giskardpy.model.collision_matrix_manager import CollisionRequest
from giskardpy.model.trajectory import Trajectory
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.utils.math import quaternion_from_rotation_matrix
from giskardpy.utils.utils import get_all_classes_in_module
from giskardpy_ros.ros2 import rospy
from giskardpy_ros.ros2.visualization_mode import VisualizationMode
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection

# from semantic_digital_twin.exceptions import SemanticAnnotationNotFoundError
from semantic_digital_twin.world_description.geometry import (
    Shape,
    Box,
    Cylinder,
    Sphere,
    Mesh,
    Color,
    Scale,
    TriangleMesh,
    FileMesh,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
    Connection,
)
from semantic_digital_twin.world_description.world_state import WorldState


# TODO probably needs some consistency check
# 1. weights are same as in message


def is_ros_message(obj: Any) -> bool:
    return hasattr(obj, "__slots__") and hasattr(obj, "get_fields_and_field_types")


# %% to ros
def to_ros_message(data):
    if isinstance(data, cas.TransformationMatrix):
        return trans_matrix_to_pose_stamped(data)
    if isinstance(data, cas.Point3):
        return point3_to_point_stamped(data)
    raise ValueError(f"{type(data)} is not a valid type")


def to_visualization_marker(data):
    if isinstance(data, Shape):
        return link_geometry_to_visualization_marker(data)


def link_to_visualization_marker(
    data: Body, mode: VisualizationMode
) -> visualization_msgs.MarkerArray:
    markers = visualization_msgs.MarkerArray()
    if mode.is_visual():
        geometries = data.visual
    else:
        geometries = data.collision
    for collision in geometries:
        if isinstance(collision, Box):
            marker = link_geometry_box_to_visualization_marker(collision)
        elif isinstance(collision, Cylinder):
            marker = link_geometry_cylinder_to_visualization_marker(collision)
        elif isinstance(collision, Sphere):
            marker = link_geometry_sphere_to_visualization_marker(collision)
        elif isinstance(collision, Mesh):
            marker = link_geometry_mesh_to_visualization_marker(collision, mode)
            if mode.is_visual():
                marker.mesh_use_embedded_materials = True
                marker.color = std_msgs.ColorRGBA()
        else:
            raise GiskardException(
                f"Can't convert {type(collision)} to visualization marker."
            )
        markers.markers.append(marker)
    return markers


def link_geometry_to_visualization_marker(data: Shape) -> visualization_msgs.Marker:
    marker = visualization_msgs.Marker()
    marker.color = color_rgba_to_ros_msg(data.color)
    marker.pose = to_ros_message(data.origin).pose
    return marker


def link_geometry_sphere_to_visualization_marker(
    data: Sphere,
) -> visualization_msgs.Marker:
    marker = link_geometry_to_visualization_marker(data)
    marker.type = visualization_msgs.Marker.SPHERE
    marker.scale.x = data.radius * 2
    marker.scale.y = data.radius * 2
    marker.scale.z = data.radius * 2
    return marker


def link_geometry_cylinder_to_visualization_marker(
    data: Cylinder,
) -> visualization_msgs.Marker:
    marker = link_geometry_to_visualization_marker(data)
    marker.type = visualization_msgs.Marker.CYLINDER
    marker.scale.x = data.width
    marker.scale.y = data.width
    marker.scale.z = data.height
    return marker


def link_geometry_box_to_visualization_marker(data: Box) -> visualization_msgs.Marker:
    marker = link_geometry_to_visualization_marker(data)
    marker.type = visualization_msgs.Marker.CUBE
    marker.scale.x = data.scale.x
    marker.scale.y = data.scale.y
    marker.scale.z = data.scale.z
    return marker


def link_geometry_mesh_to_visualization_marker(
    data: Mesh, mode: VisualizationMode
) -> visualization_msgs.Marker:
    marker = link_geometry_to_visualization_marker(data)
    marker.type = visualization_msgs.Marker.MESH_RESOURCE
    if mode.is_collision_decomposed():
        marker.mesh_resource = "file://" + data.collision_file_name_absolute
    elif isinstance(data, TriangleMesh):
        marker.mesh_resource = "file://" + data.file.name
    elif isinstance(data, FileMesh):
        marker.mesh_resource = "file://" + data.filename
    marker.scale.x = data.scale.x
    marker.scale.y = data.scale.y
    marker.scale.z = data.scale.z
    marker.mesh_use_embedded_materials = False
    return marker


def color_rgba_to_ros_msg(data: Color) -> std_msgs.ColorRGBA:
    return std_msgs.ColorRGBA(r=data.R, g=data.G, b=data.B, a=data.A)


def trans_matrix_to_pose_stamped(
    data: cas.TransformationMatrix,
) -> geometry_msgs.PoseStamped:
    pose_stamped = geometry_msgs.PoseStamped()
    pose_stamped.header.frame_id = str(data.reference_frame.name.name)
    position = data.to_position().to_np()
    orientation = data.to_rotation_matrix().to_quaternion().to_np()
    pose_stamped.pose.position = geometry_msgs.Point(
        x=position[0], y=position[1], z=position[2]
    )
    pose_stamped.pose.orientation = geometry_msgs.Quaternion(
        x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3]
    )
    return pose_stamped


def numpy_to_pose_stamped(
    data: np.ndarray, reference_frame: str
) -> geometry_msgs.PoseStamped:
    pose_stamped = geometry_msgs.PoseStamped()
    pose_stamped.header.frame_id = str(reference_frame)
    pose_stamped.pose.position.x = data[0, 3]
    pose_stamped.pose.position.y = data[1, 3]
    pose_stamped.pose.position.z = data[2, 3]
    q = quaternion_from_rotation_matrix(data)
    pose_stamped.pose.orientation = geometry_msgs.Quaternion(
        x=q[0], y=q[1], z=q[2], w=q[3]
    )
    return pose_stamped


def point3_to_point_stamped(data: cas.Point3) -> geometry_msgs.PointStamped:
    point_stamped = geometry_msgs.PointStamped()
    point_stamped.header.frame_id = str(data.reference_frame.name.name)
    position = data.to_np()
    point_stamped.point = geometry_msgs.Point(
        x=position[0], y=position[1], z=position[2]
    )
    return point_stamped


def trans_matrix_to_transform_stamped(
    data: cas.TransformationMatrix,
) -> geometry_msgs.TransformStamped:
    transform_stamped = geometry_msgs.TransformStamped()
    transform_stamped.header.frame_id = data.reference_frame.name.name
    transform_stamped.child_frame_id = data.child_frame.name.name
    position = data.to_position().to_np()
    orientation = data.to_rotation_matrix().to_quaternion().to_np()
    transform_stamped.transform.translation = geometry_msgs.Vector3(
        x=position[0], y=position[1], z=position[2]
    )
    transform_stamped.transform.rotation = geometry_msgs.Quaternion(
        x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3]
    )
    return transform_stamped


def trajectory_to_ros_trajectory(
    data: Trajectory,
    sample_period: float,
    start_time: Union[Time, float],
    joints: List[ActiveConnection],
    fill_velocity_values: bool = True,
) -> trajectory_msgs.JointTrajectory:
    if isinstance(start_time, (int, float)):
        start_time = Time(seconds=start_time)
    trajectory_msg = trajectory_msgs.JointTrajectory()
    trajectory_msg.header.stamp = start_time.to_msg()
    trajectory_msg.joint_names = []
    for time, traj_point in enumerate(data):
        p = trajectory_msgs.JointTrajectoryPoint()
        p.time_from_start = Duration(seconds=time * sample_period).to_msg()
        for joint in joints:
            for free_variable in joint.active_dofs:
                if free_variable.name in traj_point:
                    if time == 0:
                        joint_name = free_variable.name
                        if isinstance(joint_name, PrefixedName):
                            joint_name = joint_name.name
                        trajectory_msg.joint_names.append(joint_name)
                    p.positions.append(traj_point[free_variable.name].position)
                    if fill_velocity_values:
                        p.velocities.append(traj_point[free_variable.name].velocity)
                else:
                    raise NotImplementedError(
                        "generated traj does not contain all joints"
                    )
        trajectory_msg.points.append(p)
    return trajectory_msg


def world_to_tf_message(world: World, include_prefix: bool) -> tf2_msgs.TFMessage:
    tf_msg = tf2_msgs.TFMessage()
    tf = world._forward_kinematic_manager.compute_tf()
    current_time = rospy.node.get_clock().now().to_msg()
    tf_msg.transforms = create_tf_message_batch(
        len(world._forward_kinematic_manager.tf)
    )
    for i, (parent_link_name, child_link_name) in enumerate(
        world._forward_kinematic_manager.tf
    ):
        pose = tf[i]
        if not include_prefix:
            parent_link_name = parent_link_name.name
            child_link_name = child_link_name.name

        p_T_c = tf_msg.transforms[i]
        p_T_c.header.frame_id = str(parent_link_name)
        p_T_c.header.stamp = current_time
        p_T_c.child_frame_id = str(child_link_name)
        p_T_c.transform.translation.x = pose[0]
        p_T_c.transform.translation.y = pose[1]
        p_T_c.transform.translation.z = pose[2]
        p_T_c.transform.rotation.x = pose[3]
        p_T_c.transform.rotation.y = pose[4]
        p_T_c.transform.rotation.z = pose[5]
        p_T_c.transform.rotation.w = pose[6]
    return tf_msg


def json_str_to_giskard_kwargs(json_str: str, world: World) -> Dict[str, Any]:
    ros_kwargs = json_str_to_ros_kwargs(json_str)
    return ros_kwargs_to_giskard_kwargs(ros_kwargs, world)


def json_str_to_ros_kwargs(json_str: str) -> Dict[str, Any]:
    d = json.loads(json_str)
    return json_dict_to_ros_kwargs(d)


def json_dict_to_ros_kwargs(d: Any) -> Dict[str, Any]:
    if isinstance(d, str):
        try:
            d = json.loads(d)
        except JSONDecodeError:
            pass
    if isinstance(d, list):
        for i, element in enumerate(d):
            d[i] = json_dict_to_ros_kwargs(element)

    if isinstance(d, dict):
        if "type" in d:
            try:
                return SubclassJSONSerializer.from_json(d)
            except ValueError:
                pass  # it wasn't a SubclassJSONSerializer
        if "message_type" in d:
            d = convert_dictionary_to_ros_message(d)
        else:
            for key, value in d.copy().items():
                del d[key]
                d[json_dict_to_ros_kwargs(key)] = json_dict_to_ros_kwargs(value)
    return d


def ros_kwargs_to_giskard_kwargs(d: Any, world: World) -> Dict[str, Any]:
    if is_ros_message(d):
        return ros_msg_to_giskard_obj(d, world)
    elif isinstance(d, list):
        for i, element in enumerate(d):
            d[i] = ros_msg_to_giskard_obj(element, world)
            if d[i] == element:
                d[i] = ros_kwargs_to_giskard_kwargs(element, world)
    elif isinstance(d, dict):
        for key, value in d.copy().items():
            d[key] = ros_kwargs_to_giskard_kwargs(value, world)
    return d


def convert_dictionary_to_ros_message(json_data):
    # maybe somehow search for message that fits to structure of json?
    if json_data["message_type"] == msg_type_as_str(giskard_msgs.MotionStatechartNode):
        json_data["message"]["kwargs"] = json.dumps(json_data["message"]["kwargs"])
    return original_convert_dictionary_to_ros_message(
        json_data["message_type"], json_data["message"]
    )


def motion_statechart_node_to_ros_msg(
    motion_graph_node,
) -> giskard_msgs.MotionStatechartNode:
    msg = giskard_msgs.MotionStatechartNode()
    msg.name = str(motion_graph_node.name)
    msg.class_name = motion_graph_node.__class__.__name__
    msg.start_condition = motion_graph_node.start_condition
    msg.pause_condition = motion_graph_node.pause_condition
    msg.end_condition = motion_graph_node.end_condition
    msg.reset_condition = motion_graph_node.reset_condition
    return msg


def exception_to_error_msg(exception: Exception) -> giskard_msgs.GiskardError:
    error = giskard_msgs.GiskardError()
    error.type = exception.__class__.__name__
    error.msg = str(exception)
    return error


# %% from ros

exception_classes = get_all_classes_in_module(
    module_name="giskardpy.data_types.exceptions", parent_class=GiskardException
)

# add all base exceptions
exception_classes.update(
    {
        name: getattr(builtins, name)
        for name in dir(builtins)
        if isinstance(getattr(builtins, name), type)
        and issubclass(getattr(builtins, name), BaseException)
    }
)


def error_msg_to_exception(msg: giskard_msgs.GiskardError) -> Optional[Exception]:
    if msg.type == giskard_msgs.GiskardError.SUCCESS:
        return None
    if msg.type in exception_classes:
        return exception_classes[msg.type](msg.msg)
    return Exception(f"{msg.type}: {msg.msg}")


def link_name_msg_to_body(
    msg: giskard_msgs.LinkName, world: World
) -> KinematicStructureEntity:
    if msg.group_name == "" and msg.name == "":
        return world.root
    if msg.group_name == "":
        return world.get_kinematic_structure_entity_by_name(msg.name)
    return world.get_kinematic_structure_entity_by_name(
        PrefixedName(msg.name, msg.group_name)
    )


def joint_name_msg_to_prefix_name(
    msg: giskard_msgs.LinkName, world: World
) -> PrefixedName:
    return world.get_connection_by_name(PrefixedName(msg.name, msg.group_name)).name


def replace_prefix_name_with_str(d: dict) -> dict:
    new_d = d.copy()
    for k, v in d.items():
        if isinstance(k, PrefixedName):
            del new_d[k]
            new_d[str(k)] = v
        if isinstance(v, PrefixedName):
            new_d[k] = str(v)
        if isinstance(v, dict):
            new_d[k] = replace_prefix_name_with_str(v)
    return new_d


def kwargs_to_json(kwargs: Dict[str, Any]) -> str:
    for k, v in kwargs.copy().items():
        if v is None:
            del kwargs[k]
        else:
            kwargs[k] = thing_to_json(v)
    kwargs = replace_prefix_name_with_str(kwargs)
    return json.dumps(kwargs)


def thing_to_json(thing: Any) -> Any:
    if isinstance(thing, list):
        return [thing_to_json(x) for x in thing]
    if isinstance(thing, dict):
        return {thing_to_json(k): thing_to_json(v) for k, v in thing.items()}
    if isinstance(thing, giskard_msgs.MotionStatechartNode):
        return thing_to_json(convert_ros_message_to_dictionary(thing))
    if is_ros_message(thing):
        return convert_ros_message_to_dictionary(thing)
    if isinstance(thing, SubclassJSONSerializer):
        return json.dumps(thing.to_json())
    return thing


def convert_ros_message_to_dictionary(message) -> dict:
    if isinstance(message, list):
        for i, element in enumerate(message):
            message[i] = convert_ros_message_to_dictionary(element)
    elif isinstance(message, dict):
        for k, v in message.copy().items():
            message[k] = convert_ros_message_to_dictionary(v)

    elif isinstance(message, tuple):
        list_values = list(message)
        for i, element in enumerate(list_values):
            list_values[i] = convert_ros_message_to_dictionary(element)
        message = tuple(list_values)

    elif is_ros_message(message):

        type_str_parts = str(type(message)).split(".")
        part1 = type_str_parts[0].split("'")[1]
        part2 = type_str_parts[-1].split("'")[0]
        message_type = f"{part1}/{part2}"
        d = {
            "message_type": message_type,
            "message": original_convert_ros_message_to_dictionary(message),
        }
        return d

    return message


def msg_type_as_str(msg_type) -> str:
    type_str_parts = str(type(msg_type())).split(".")
    part1 = type_str_parts[0].split("'")[1]
    part2 = type_str_parts[1]
    part3 = type_str_parts[-1].split("'")[0]
    return f"{part1}/{part2}/{part3}"


def ros_msg_to_giskard_obj(msg, world: World):
    if isinstance(msg, sensor_msgs.JointState):
        return ros_joint_state_to_giskard_joint_state(msg)
    elif isinstance(msg, geometry_msgs.PoseStamped):
        return pose_stamped_to_trans_matrix(msg, world)
    elif isinstance(msg, geometry_msgs.Pose):
        return pose_to_trans_matrix(msg)
    elif isinstance(msg, geometry_msgs.PointStamped):
        return point_stamped_to_point3(msg, world)
    elif isinstance(msg, geometry_msgs.Vector3Stamped):
        return vector_stamped_to_vector3(msg, world)
    elif isinstance(msg, geometry_msgs.QuaternionStamped):
        return quaternion_stamped_to_quaternion(msg, world)
    elif isinstance(msg, giskard_msgs.CollisionEntry):
        return collision_entry_msg_to_giskard(msg, world=world)
    elif isinstance(msg, giskard_msgs.GiskardError):
        return error_msg_to_exception(msg)
    return msg


def convert_prefixed_name(
    node_class: Type[MotionStatechartNode],
    kwargs: Dict[str, Any],
    world: World,
) -> Dict[str, Any]:
    for field in fields(node_class):
        if field.name in kwargs:
            value = kwargs[field.name]
            new_value = replace_connection_and_kinematic_structure_entity_in_kwargs(
                field.type, value, world
            )
            kwargs[field.name] = new_value
    return kwargs


def replace_connection_and_kinematic_structure_entity_in_kwargs(
    type_hint: Any, kwargs_value: Any, world: World
) -> Any:
    if isinstance(kwargs_value, dict):
        type_args = get_args(type_hint)
        for key, value in kwargs_value.copy().items():
            replaced_key = replace_str_prefixed_name(type_args[0], key, world)
            if replaced_key is not None:
                del kwargs_value[key]
                kwargs_value[replaced_key] = value
            replaced_value = replace_str_prefixed_name(type_args[1], value, world)
            if replaced_value is not None:
                kwargs_value[key] = replaced_value
        return kwargs_value
    if isinstance(kwargs_value, list):
        type_args = get_args(type_hint)
        for index, value in enumerate(kwargs_value):
            replaced_value = replace_str_prefixed_name(type_args[0], value, world)
            if replaced_value is not None:
                kwargs_value[index] = replaced_value
        return kwargs_value
    replaced_value = replace_str_prefixed_name(
        get_origin(type_hint) or type_hint, kwargs_value, world
    )
    if replaced_value is not None:
        return replaced_value
    return kwargs_value


def replace_str_prefixed_name(
    kwargs_type, kwargs_value, world: World
) -> Optional[Union[KinematicStructureEntity, Connection]]:
    if isinstance(kwargs_value, (str, PrefixedName)):
        if issubclass(kwargs_type, KinematicStructureEntity):
            return world.get_kinematic_structure_entity_by_name(kwargs_value)
        elif issubclass(kwargs_type, Connection):
            return world.get_connection_by_name(kwargs_value)


def ros_joint_state_to_giskard_joint_state(
    msg: sensor_msgs.JointState, prefix: Optional[str] = None
) -> WorldState:
    js = WorldState()
    for i, joint_name in enumerate(msg.name):
        joint_name = PrefixedName(joint_name, prefix)
        js[joint_name][Derivatives.position] = msg.position[i]
    return js


def world_body_to_geometry(msg: giskard_msgs.WorldBody, color: Color) -> Shape:
    if msg.type == giskard_msgs.WorldBody.URDF_BODY:
        raise NotImplementedError()
    elif msg.type == giskard_msgs.WorldBody.PRIMITIVE_BODY:
        if msg.shape.type == msg.shape.BOX:
            scale = Scale(
                msg.shape.dimensions[msg.shape.BOX_X],
                msg.shape.dimensions[msg.shape.BOX_Y],
                msg.shape.dimensions[msg.shape.BOX_Z],
            )
            geometry = Box(origin=cas.TransformationMatrix(), scale=scale, color=color)
        elif msg.shape.type == msg.shape.CYLINDER:
            geometry = Cylinder(
                origin=cas.TransformationMatrix(),
                height=msg.shape.dimensions[msg.shape.CYLINDER_HEIGHT],
                width=msg.shape.dimensions[msg.shape.CYLINDER_RADIUS] * 2,
                color=color,
            )
        elif msg.shape.type == msg.shape.SPHERE:
            geometry = Sphere(
                origin=cas.TransformationMatrix(),
                radius=msg.shape.dimensions[msg.shape.SPHERE_RADIUS],
                color=color,
            )
        else:
            raise CorruptShapeException(
                f"Primitive shape of type {msg.shape.type} not supported."
            )
    elif msg.type == giskard_msgs.WorldBody.MESH_BODY:
        if msg.scale.x == 0 or msg.scale.y == 0 or msg.scale.z == 0:
            raise CorruptShapeException(f"Scale of mesh contains 0: {msg.scale}")
        geometry = Mesh(
            origin=cas.TransformationMatrix(),
            filename=msg.mesh,
            scale=Scale(msg.scale.x, msg.scale.y, msg.scale.z),
            color=color,
        )
    else:
        raise CorruptShapeException(f"World body type {msg.type} not supported")
    return geometry


def pose_stamped_to_trans_matrix(
    msg: geometry_msgs.PoseStamped, world: World
) -> cas.TransformationMatrix:
    p = cas.Point3(
        x_init=msg.pose.position.x,
        y_init=msg.pose.position.y,
        z_init=msg.pose.position.z,
    )
    R = cas.Quaternion(
        x_init=msg.pose.orientation.x,
        y_init=msg.pose.orientation.y,
        z_init=msg.pose.orientation.z,
        w_init=msg.pose.orientation.w,
    ).to_rotation_matrix()
    result = cas.TransformationMatrix.from_point_rotation_matrix(
        point=p,
        rotation_matrix=R,
        reference_frame=world.get_kinematic_structure_entity_by_name(
            msg.header.frame_id
        ),
    )
    return result


def pose_to_trans_matrix(msg: geometry_msgs.Pose) -> cas.TransformationMatrix:
    p = cas.Point3(msg.position.x, msg.position.y, msg.position.z)
    R = cas.Quaternion(
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
    ).to_rotation_matrix()
    result = cas.TransformationMatrix.from_point_rotation_matrix(
        point=p, rotation_matrix=R, reference_frame=None
    )
    return result


def point_stamped_to_point3(
    msg: geometry_msgs.PointStamped, world: World
) -> cas.Point3:
    return cas.Point3(
        msg.point.x,
        msg.point.y,
        msg.point.z,
        reference_frame=world.get_kinematic_structure_entity_by_name(
            msg.header.frame_id
        ),
    )


def vector_stamped_to_vector3(
    msg: geometry_msgs.Vector3Stamped, world: World
) -> cas.Vector3:
    return cas.Vector3(
        msg.vector.x,
        msg.vector.y,
        msg.vector.z,
        reference_frame=world.get_kinematic_structure_entity_by_name(
            msg.header.frame_id
        ),
    )


def quaternion_stamped_to_quaternion(
    msg: geometry_msgs.QuaternionStamped, world: World
) -> cas.RotationMatrix:
    return cas.Quaternion(
        msg.quaternion.x,
        msg.quaternion.y,
        msg.quaternion.z,
        msg.quaternion.w,
        reference_frame=world.get_kinematic_structure_entity_by_name(
            msg.header.frame_id
        ),
    ).to_rotation_matrix()


def collision_entry_msg_to_giskard(
    msg: giskard_msgs.CollisionEntry, world: World
) -> CollisionRequest:
    if msg.distance == -1:
        distance = None
    else:
        distance = msg.distance

    try:
        view1 = world.get_semantic_annotation_by_name(msg.group1)
    except WorldEntityNotFoundError as e:
        view1 = None

    try:
        view2 = world.get_semantic_annotation_by_name(msg.group2)
    except WorldEntityNotFoundError as e:
        view2 = None

    return CollisionRequest(
        type_=msg.type,
        distance=distance,
        semantic_annotation1=view1,
        semantic_annotation2=view2,
    )


__tf_messages: List[TransformStamped] = None


def __create_tf_messages():
    global __tf_messages
    __tf_messages = [TransformStamped() for _ in range(10000)]


threading.Thread(target=__create_tf_messages, daemon=True).start()


def create_tf_message_batch(size: int) -> List[TransformStamped]:
    global __tf_messages
    while __tf_messages is None:
        sleep(0.01)
    return __tf_messages[:size]
