import gymnasium as gym
import numpy as np
from .image_processing import *  #Adapt to your path
from .distance import *

class ImageEnv(gym.Wrapper):
    """Wrapper to modify the gymnasium environment"""
    def __init__(
        self,
        env,
        skip_frames=8,
        stack_frames=4,
        initial_no_op=50,
        **kwargs
    ):
        super(ImageEnv, self).__init__(env, **kwargs)
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.initial_no_op=initial_no_op
    
    def reset(self):
        # Reset the original environment.
        s, info = self.env.reset()

        # Do nothing for the next `self.initial_no_op` steps
        for i in range(self.initial_no_op):
            s, r, terminated, truncated, info = self.env.step(0)
        
        # Convert a frame to 84 X 84 gray scale one
        s = preprocess(s)

        # The initial observation is simply a copy of the frame `s`
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))  # [4, 84, 84]
        return self.stacked_state
    
    def step(self, action):
        # We take an action for self.skip_frames steps
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)            
            if terminated or truncated:
                break        
        
        s = preprocess(s)
        #if distance(s)<=-74 or terminated or truncated:
        if terminated or truncated:
            over=True
        else:
            over=False

        # Push the current frame `s` at the end of self.stacked_state
        self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)

        return self.stacked_state, over
