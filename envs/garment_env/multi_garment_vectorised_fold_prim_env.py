import numpy as np
import gym
from .multi_garment_env import MultiGarmentEnv


# @ray.remote
class MultiGarmentVectorisedFoldPrimEnv(MultiGarmentEnv):
    
    def __init__(self, task, config):
        super().__init__(task, config)
        self.action_space = gym.spaces.Box(-1, 1, (8, ), dtype=np.float32)

    def step(self, action): ## get action for hybrid action primitive, action defined in the observation space
        self.last_info = self.info
        self.evaluate_result = None
        self.overstretch = 0
        dict_action = {
            'norm-pixel-fold': {
                'pick_0': action[:2],
                'pick_1': action[2:4],
                'place_0': action[4:6],
                'place_1': action[6:8]
            }
        }

        info = self.action_tool.step(self, dict_action)
        self.action_step += 1
        # print('step', self.action_step)
        self.info = self._process_info(info)
        dict_applied_action = self.info['applied_action']
        vector_action = []
        for param_name in ['pick_0', 'pick_1', 'place_0', 'place_1']:
            vector_action.append(dict_action['norm-pixel-fold'][param_name])
        #print('vector_action', vector_action)
        vector_action = np.stack(vector_action).flatten()
        self.info['applied_action'] = vector_action
        
        obs, reward, done = self.info['observation'], self.info['reward'], self.info['done']
        
        obs_ = {}
        obs_['image'] = obs['image']
        obs_['is_first'] = False
        obs_['is_terminal'] = done        
        #print('reward', info['reward'])
        return obs_, reward, done, self.info