import retro
import imageio
import numpy as np
import sys
import av
import os
sys.path.append("main")
from common.retro_wrappers import *
from common.const import *
from PIL import Image

def make_env(side, state_path=None):
    players = 2
    env = retro.make(
        game="StreetFighterIISpecialChampionEdition-Genesis",
        state="stars/Champion.Level1.RyuVsRyu.left_star1.state",
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        players=players,
    )
    env = SFWrapper(env, init_level=1, side=side, reset_type="round", enable_combo=False, null_combo=False, transform_action=False, verbose=True)
    return env

sides = ["both"]
for side in sides:
    try:
        env.close()
    except:
        pass
    env = make_env(side)
    env.reset()
    print(env.buttons)
    print(env.action_space)
    video_log = [Image.fromarray(env.render(mode="rgb_array"))]
    #display(video_log[-1])
    while True:
    # for _ in range(100):
        _obs, _rew, _rew_other, done, _info = env.step(np.hstack([env.action_space.sample(), env.action_space.sample()]))
        video_log.append(Image.fromarray(env.render(mode="rgb_array")))
        if done or len(video_log) % 50 == 0:
            print('episode_length: ', len(video_log))
            print(_info)
            #display(video_log[-1])
            if done:
                break
    print('episode_length: ', len(video_log))
    print(_info)

    imageio.mimsave(f'./sf2_{0}.gif', [np.array(img) for img in video_log], fps=10)

    env.close()