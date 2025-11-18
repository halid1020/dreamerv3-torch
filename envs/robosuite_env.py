import numpy as np
import gym
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper

class Robosuite:
    """DMC-style wrapper for Robosuite."""

    metadata = {}

    def __init__(
        self,
        task,
        robots,
        controller_name="OSC_POSE",
        action_repeat=1,
        size=(256, 256),
        #camera="frontview",
        horizon=500,
        use_camera_obs=False,
        seed=0,
    ):
        """
        Parameters mimic DeepMindControl style:
            name: robosuite task name (e.g., "Lift", "PickPlace")
            robots: robot list (e.g., ["Panda"])
            controller_name: robosuite controller
            action_repeat: how many simulation steps per agent step
            size: camera observation size (H, W)
            camera: robosuite camera name
            horizon: episode length
            use_camera_obs: include rendered RGB image observation
            seed: deterministic seed
        """

        # -----------------------
        # Make Robosuite environment
        # -----------------------
        #print('Making Robosuite Env')
        #print(task, robots, horizon, controller_name)
        self._env = GymWrapper(suite.make(
            env_name=task,
            robots=robots,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_object_obs=True,
            use_camera_obs=True,
            reward_shaping=True,
            control_freq=20,
            horizon=horizon,
            controller_configs=suite.load_controller_config(
                default_controller=controller_name
            )
        ),)
        #print('Finished Robosuite Env')
        self.seed = seed
        # Config
        self._action_repeat = action_repeat
        self._horizon = horizon
        self._size = size
        self._use_camera_obs = use_camera_obs
        self._step_count = 0

        self.reward_range = [-np.inf, np.inf]

        # -----------------------
        # Define DMC-style spaces
        # -----------------------
        self.action_space = self._env.action_space
        #self.observation_space = self._env.observation_space
        #print('Built Robosuite Env')

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if len(value.shape) == 0:
                shape = (1,)
            else:
                shape = value.shape
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)
    
    # @property
    # def action_space(self):
    #     spec = self._env.action_spec()
    #     return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    
    # =====================================================================
    #   RESET
    # =====================================================================
    def reset(self):
        obs, _ = self._env.reset(seed=self.seed)
        self._step_count = 0

        obs = self._convert_obs(obs, is_first=True, is_terminal=False)
        #print('reset obs', obs)
        return obs

    # =====================================================================
    #   STEP (DMC-style)
    # =====================================================================
    def step(self, action):
        assert np.isfinite(action).all(), action

        total_reward = 0
        done = False
        info = {}

        for _ in range(self._action_repeat):
            obs, reward, terminated, truncated, env_info = self._env.step(action)
            total_reward += reward
            self._step_count += 1

            if terminated or truncated:
                done = True
                break

        is_terminal = done
        is_first = False

        obs_dict = self._convert_obs(obs, is_first=is_first, is_terminal=is_terminal)
        info = {"discount": np.array(1.0, np.float32)}
        return obs_dict, total_reward, done, info

    # =====================================================================
    #   OBSERVATION CONVERSION (DMC-format)
    # =====================================================================
    def _convert_obs(self, obs, is_first, is_terminal):
        """Convert Robosuite observation dict → DMC-style flat dict."""
        result = {}

        # Copy robot and object states
        # for key, val in obs.items():
        #     if key.startswith("robot") or key.startswith("object"):
        #         result[key] = val.astype(np.float32)

        # # Camera if used
        # if self._use_camera_obs:
        #     # Robosuite GymWrapper returns image under "<camera>_image"
        #     img_key = f"{self._camera}_image"
        
        result["image"] = self.render()

        # Episode flags
        result["is_first"] = is_first
        result["is_terminal"] = is_terminal

        return result

    # =====================================================================
    #   RENDER
    # =====================================================================
    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only rgb_array mode is supported.")

        frame = self._env.sim.render(
            camera_name="frontview",
            width=self._size[1],
            height=self._size[0],
        )
        frame = np.flipud(frame)
        return frame

def main():
    # ---------------------------------------
    # Create the wrapped Robosuite environment
    # ---------------------------------------
    env = Robosuite(
        task="Lift",
        robots="Panda",
        controller_name="OSC_POSE",
        action_repeat=1,
        size=(256, 256),
        horizon=200,
        use_camera_obs=False,
        seed=0,
    )

    # ---------------------------------------
    # Reset environment
    # ---------------------------------------
    obs = env.reset()
    print("Initial observation keys:", obs.keys())
    print("Initial image shape:", obs["image"].shape)
    print('Initial Obs keys', obs.keys() )

    # ---------------------------------------
    # Step through the environment
    # ---------------------------------------
    num_steps = 10
    for step in range(num_steps):
        # Sample a random action
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action)

        print(f"\nStep {step}")
        print("  Reward:", reward)
        print("  Done:", done)
        print("  Info:", info)
        print("  Obs keys:", obs.keys())
        print("  Image shape:", obs["image"].shape)
        print("  Is Terminal", obs['is_terminal'])
        print("  Is First", obs["is_first"])

        if done:
            print("Episode terminated early. Resetting...")
            obs = env.reset()

    print("\nFinished running test episode.")

    # ---------------------------------------
    # Example render call (1 frame)
    # ---------------------------------------
    frame = env.render()
    print("Rendered frame shape:", frame.shape)


if __name__ == "__main__":
    main()
