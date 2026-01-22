from typing import Optional

from py_trees.composites import Sequence

from giskardpy_ros.tree.behaviors.debug_marker_publisher import DebugMarkerPublisher
from giskardpy_ros.tree.behaviors.publish_debug_expressions import (
    PublishDebugExpressions,
    QPDataPublisherConfig,
)
from giskardpy_ros.tree.behaviors.publish_feedback import PublishFeedback
from giskardpy_ros.tree.behaviors.publish_joint_states import PublishWorldState
from giskardpy_ros.tree.behaviors.tf_publisher import TfPublishingModes, TFPublisher


class PublishState(Sequence):
    debug_marker_publisher: Optional[DebugMarkerPublisher]

    def __init__(self, name: str = "publish state"):
        super().__init__(name, memory=True)
        self.debug_marker_publisher = None

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

    def add_joint_state_publisher(self):
        node = PublishWorldState()
        self.add_child(node)
