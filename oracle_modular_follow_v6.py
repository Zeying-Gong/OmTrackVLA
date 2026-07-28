"""Version 6 policy settings for the modular oracle follower."""

from oracle_modular_follow import OracleNavmeshFollower


class OracleNavmeshFollowerV6(OracleNavmeshFollower):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("tracking_distance_m", 1.3)
        kwargs.setdefault("prioritize_visibility", True)
        kwargs.setdefault("incoming_safe_retreat_distance_m", 2.2)
        kwargs.setdefault("emergency_safe_retreat_distance_m", 0.85)
        kwargs.setdefault("tracking_mask_max_pixels", int(384 * 384 * 0.3))
        kwargs.setdefault("visibility_reframe_after_steps", 8)
        kwargs.setdefault("coordinate_approach_min_scale", 0.5)
        kwargs.setdefault("incoming_memory_steps", 6)
        kwargs.setdefault("evasion_start_distance_m", 2.2)
        super().__init__(*args, **kwargs)
