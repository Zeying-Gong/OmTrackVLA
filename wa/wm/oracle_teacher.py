"""Teacher-only bridge to the existing continuous NavMesh Oracle.

No simulator stepping/resetting or alternate physics is performed here.
The benchmark owns execution. This object must never be used as WA inputs.
"""
import numpy as np

class OracleTeacher:
    def __init__(self, module=None, controller_kwargs=None):
        if module is None:
            import oracle_modular_follow as module
        self.module = module
        self.perception = module.OraclePerception()
        self.controller = module.OracleNavmeshFollower(**(controller_kwargs or {}))
        self.environment = None
        self.last_trajectory = None
        self.reply_error = None
        self.last_decision = None

    def bind_environment(self, env):
        self.environment = env

    def reset(self, *args, **kwargs):
        self.perception.reset()
        self.controller.reset()
        self.last_trajectory = None
        self.last_decision = None
        self.reply_error = None

    def act(self, observations, detector, episode_id, instruction=None):
        del detector, instruction
        if self.environment is None:
            raise RuntimeError("Oracle requires benchmark environment binding")
        env = self.environment
        if str(env.current_episode.episode_id) != str(episode_id):
            raise RuntimeError("Oracle episode mismatch")
        sim = env.sim
        robot = sim.agents_mgr[1].articulated_agent
        human = sim.agents_mgr[0].articulated_agent
        m = self.module
        target = self.perception(
            observations[m.RGB_KEY], observations[m.PANOPTIC_KEY],
            int(env.current_episode.info["main_human_semantic_id"]),
            m.local_target(robot, human),
        )
        decision = self.controller(sim, robot, human, target)
        action = np.asarray(decision.action.as_habitat(), dtype=float)
        if action.shape != (3,) or not np.isfinite(action).all() or (abs(action) > 1).any():
            raise RuntimeError("Oracle returned invalid normalized control")
        self.last_decision = decision
        return action.tolist()

    def close(self):
        pass
