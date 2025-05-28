from stable_baselines3.sac.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, MLPAACCNNPolicy
from stable_baselines3.sac.sac import SAC
#from stable_baselines3.sac.magics_CL import MAGICS_CL
from stable_baselines3.sac.magics_AL import MAGICS_AL

__all__ = ["MLPAACCNNPolicy", "CnnPolicy", "MlpPolicy", "MultiInputPolicy", "SAC", "MAGICS_CL", "MAGICS_AL"]
