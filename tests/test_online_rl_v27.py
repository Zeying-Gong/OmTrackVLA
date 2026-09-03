import numpy as np

from hybrid_flux.online_rl_utils_v27 import tracking_reward_v27


def test_sticky_legacy_collision_is_charged_once():
    previous = {
        "distance_to_leader": 2.0,
        "human_following": 1.0,
        "human_collision": 1.0,
        "collision_metric_available": 0.0,
    }
    current = dict(previous)
    reward, components = tracking_reward_v27(
        previous, current, np.zeros(3, dtype=np.float32), False
    )
    assert components["collision"] == 0.0
    assert reward > 0.0


def test_proper_collision_metric_is_an_event():
    previous = {
        "distance_to_leader": 0.7,
        "human_following": 0.0,
        "did_multi_agents_collide": 0.0,
        "collision_metric_available": 1.0,
    }
    current = dict(previous)
    current["did_multi_agents_collide"] = 1.0
    _, components = tracking_reward_v27(
        previous, current, np.zeros(3, dtype=np.float32), False
    )
    assert components["collision"] == 1.0

    _, components_again = tracking_reward_v27(
        current, current, np.zeros(3, dtype=np.float32), False
    )
    assert components_again["collision"] == 0.0


def test_recovery_transition_has_positive_signal():
    previous = {
        "distance_to_leader": 4.0,
        "human_following": 0.0,
        "did_multi_agents_collide": 0.0,
        "collision_metric_available": 1.0,
    }
    current = dict(previous)
    current.update({"distance_to_leader": 2.0, "human_following": 1.0})
    reward, components = tracking_reward_v27(
        previous, current, np.zeros(3, dtype=np.float32), False
    )
    assert components["recovered"] == 1.0
    assert reward > 0.0
