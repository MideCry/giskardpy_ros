from py_trees.common import Status

from giskardpy.data_types.exceptions import SelfCollisionViolatedException
from giskardpy.model.collision_world_syncer import Collisions
from giskardpy.utils.decorators import record_time
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy_ros.tree.blackboard_utils import (
    catch_and_raise_to_blackboard,
    GiskardBlackboard,
)


class CollisionChecker(GiskardBehavior):

    def __init__(self, name: str):
        super().__init__(name)

    # def initialise(self) -> None:
    #     god_map.collision_scene.add_added_checks()
    #     super().initialise()

    def are_self_collisions_violated(self, collsions: Collisions) -> None:
        for key, self_collisions in collsions.self_collisions.items():
            for self_collision in self_collisions[
                :-1
            ]:  # the last collision is always some default crap
                if self_collision.link_b_hash == 0:
                    continue  # Fixme figure out why there are sometimes two default collision entries
                distance = self_collision.contact_distance
                if distance < 0.0:
                    raise SelfCollisionViolatedException(
                        f"{self_collision.original_body_a} and "
                        f"{self_collision.original_body_b} violate distance threshold:"
                        f"{self_collision.contact_distance} < {0}"
                    )

    @catch_and_raise_to_blackboard(skip_on_exception=False)
    @record_time
    def update(self) -> Status:
        """
        Computes closest point info for all robot links and safes it to the god map.
        """
        # collisions = GiskardBlackboard().executor.collision_scene.check_collisions()
        # self.are_self_collisions_violated(collisions)
        return Status.RUNNING
