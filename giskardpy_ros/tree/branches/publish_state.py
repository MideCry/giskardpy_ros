from typing import Optional

from py_trees.composites import Sequence

from giskardpy.utils.decorators import toggle_on, toggle_off
from giskardpy_ros.ros2.ros_msg_visualization import VisualizationMode
from giskardpy_ros.tree.behaviors.debug_marker_publisher import DebugMarkerPublisher
from giskardpy_ros.tree.behaviors.publish_debug_expressions import (
    PublishDebugExpressions,
    QPDataPublisherConfig,
)
from giskardpy_ros.tree.behaviors.publish_feedback import PublishFeedback
from giskardpy_ros.tree.behaviors.publish_joint_states import PublishJointState
from giskardpy_ros.tree.behaviors.tf_publisher import TfPublishingModes, TFPublisher
from giskardpy_ros.tree.behaviors.visualization import VisualizationBehavior


class PublishState(Sequence):
    visualization_behavior: Optional[VisualizationBehavior]
    debug_marker_publisher: Optional[DebugMarkerPublisher]

    def __init__(self, name: str = "publish state"):
        super().__init__(name, memory=True)
        self.visualization_behavior = None
        self.debug_marker_publisher = None

    @toggle_on("visualization_marker_behavior")
    def add_visualization_marker_behavior(
        self, mode: VisualizationMode, scale_scale: float = 1.0
    ):
        if self.visualization_behavior is None:
            self.visualization_behavior = VisualizationBehavior(
                mode, scale_scale=scale_scale
            )
        self.add_child(self.visualization_behavior)

    @toggle_off("visualization_marker_behavior")
    def remove_visualization_marker_behavior(self):
        self.remove_child(self.visualization_behavior)

    def add_debug_marker_publisher(self):
        self.debug_marker_publisher = DebugMarkerPublisher()
        self.add_child(self.debug_marker_publisher)

    def add_publish_feedback(self):
        self.add_child(PublishFeedback())

    def add_tf_publisher(
        self,
        include_prefix: bool = False,
        tf_topic: str = "tf",
        mode: TfPublishingModes = TfPublishingModes.attached_and_world_objects,
    ):
        node = TFPublisher(
            "publish tf", mode=mode, tf_topic=tf_topic, include_prefix=include_prefix
        )
        self.add_child(node)

    def add_qp_data_publisher(self, publish_config: QPDataPublisherConfig):
        node = PublishDebugExpressions(publish_config=publish_config)
        self.add_child(node)

    def add_joint_state_publisher(
        self,
        topic_name: Optional[str] = None,
        include_prefix: bool = False,
        only_prismatic_and_revolute: bool = True,
    ):
        node = PublishJointState(
            include_prefix=include_prefix,
            topic_name=topic_name,
            only_prismatic_and_revolute=only_prismatic_and_revolute,
        )
        self.add_child(node)
