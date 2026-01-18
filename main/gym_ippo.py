import gym
import gymnasium
from gymnasium.envs.registration import register
from common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR
import wandb
import argparse
from stable_baselines3.common.adversarial_envs.my_pendulum import my_PendulumEnv
from stable_baselines3.common.adversarial_envs.my_walker2d_v4 import my_Walker2dEnv
from stable_baselines3.common.adversarial_envs.my_mountain_car_continuous import my_Continuous_MountainCarEnv
from stable_baselines3.common.adversarial_envs.my_half_cheetah import my_HalfCheetahEnv
from stable_baselines3.common.adversarial_envs.my_hopper_v5 import my_HopperEnv
from stable_baselines3.common.adversarial_envs.my_ant_v5 import my_AntEnv
def critic_decay_schedule(initial_value):
    return lambda progress: initial_value * (1 - progress)
def actor_decay_schedule(initial_value):
    return lambda progress: initial_value * (1 - progress)
register(
    # unique identifier for the env `name-version`
    id="my_pendulum",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=my_PendulumEnv,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=200,
)

env = gymnasium.make("my_pendulum")
def env_generator(max_envs: int = 0, i_start: int = 0, j_start: int = 0, STATE=None):
    env_name = STATE[0].split("_vs_")[1]
    return gym.make(env_name)
PLAYER = "ego0"
OPPONENT_LIST = ["adv0"]

def main(args):
    env_name = args.env_name
    STATE = ["Champion.%s.%sVs%s.2Player.state" % (env_name, PLAYER, OPPONENT_LIST[0])]
    state_list = STATE
    finetune_model = CleanDerivativeFreeSPAR(
            policy="AACCnnPolicy",
            env=env,
            device="cuda",
            c_learning_rate=args.c_lr,
            d_learning_rate=args.d_lr,
            v_learning_rate=args.v_lr,
            verbose=2,
            n_steps=args.num_env_steps,
            batch_size=300,
            n_epochs=4,
            state_list=state_list,
            envs_per_matchup=1,
            env_generator_func=env_generator,
            num_adversaries=1,
            n_env_per_adv=1,
            seed= 0,
            target_kl=None,
            use_mirror=False,
            use_lr_annealing=args.use_lr_annealing,
            lr_anneal_coeff=args.lr_anneal_coeff
        )
    finetune_model.learn(total_timesteps=args.num_env_steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, choices=["my_pendulum", "my_walker2d", "my_mountain_car", "my_half_cheetah", "my_hopper", "my_ant"], required=True, default="my_pendulum")
    parser.add_argument("--c_lr", type=float, required=True, default=1e-6)
    parser.add_argument("--d_lr", type=float, required=True, default=2e-6)
    parser.add_argument("--v_lr", type=float, required=True, default=4e-6)
    parser.add_argument("--num_perturbs", type=int, required=True, default=10)
    parser.add_argument("--load_path", type=str, required=True, default=None)
    parser.add_argument("--continue_training", type=bool, required=True, default=False)
    parser.add_argument("--model-file", type=str, required=True, default=None)
    parser.add_argument("--use_lr_annealing", type=bool, required=True, default=False)
    parser.add_argument("--lr_anneal_coeff", type=float, required=True, default=0.995)
    parser.add_argument("--checkpoint_interval", type=int, required=True, default=100000)
    parser.add_argument("--num_env_steps", type=int, required=True, default=1024)
    parser.add_argument("--envs_per_matchup", type=int, required=True, default=1)
    args = parser.parse_args()
    main(args)