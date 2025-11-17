from py_trees.common import Status
from py_trees.composites import Sequence

from giskardpy_ros.tree.behaviors.collision_scene_updater import CollisionSceneUpdater
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.behaviors.world_updater import ProcessWorldUpdate
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard
from giskardpy_ros.tree.branches.publish_state import PublishState
from giskardpy_ros.tree.branches.synchronization import Synchronization


class HasWorldUpdate(GiskardBehavior):
    def __init__(self):
        super().__init__("has world update?")

    def update(self) -> Status:
        if len(GiskardBlackboard().giskard.model_synchronizer.missed_messages) > 0:
            return Status.SUCCESS
        return Status.FAILURE


class UpdateWorld(Sequence):
    synchronization: Synchronization
    publish_state: PublishState
    goal_received: HasWorldUpdate
    process_goal: ProcessWorldUpdate

    def __init__(self):
        name = "update world"
        super().__init__(name, memory=True)
        self.goal_received = HasWorldUpdate()
        self.process_goal = ProcessWorldUpdate()

        self.add_child(self.goal_received)
        self.add_child(self.process_goal)
        # self.add_child(NotifyModelChange())
        self.add_child(CollisionSceneUpdater())
