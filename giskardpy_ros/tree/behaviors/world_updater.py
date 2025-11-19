from threading import Thread

from py_trees.common import Status

from giskardpy.middleware import get_middleware
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard


class ProcessWorldUpdate(GiskardBehavior):

    def __init__(self):
        name = f"Processing world updates"
        self.worker_thread = None
        super().__init__(name)

    def update(self) -> Status:
        if self.worker_thread is None:
            self.worker_thread = Thread(target=self.process_goal, name=self.name)
            self.worker_thread.start()
        else:
            if not self.worker_thread.is_alive():
                self.worker_thread = None
                get_middleware().loginfo(
                    f"Finished world update, model version: {GiskardBlackboard().executor.world._model_manager.version}."
                )
                return Status.SUCCESS
        return Status.RUNNING

    def process_goal(self):
        GiskardBlackboard().giskard.model_synchronizer.apply_missed_messages()
