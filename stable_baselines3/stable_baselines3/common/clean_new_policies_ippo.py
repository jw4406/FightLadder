import torch as th
import torch.nn as nn
import itertools

from stable_baselines3.common.clean_new_policies import CleanActorActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs


class CleanIPPOActorActorCriticPolicy(CleanActorActorCriticPolicy):
    """
    CleanActorActorCriticPolicy + dedicated ego value head.
    Adversary heads and existing value path remain unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ego_vf_features_extractor = self.make_features_extractor()
        self.ego_value_net = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_vf, 256),
            self.activation_fn(),
            nn.Linear(256, 256),
            self.activation_fn(),
            nn.Linear(256, 1),
        ).to(self.device)
        # Keep optimizer setup close to original behavior while adding ego value params.
        self.ego_value_optimizer = self.optimizer_class(
            itertools.chain(self.mlp_extractor.ego_value_net.parameters(), self.ego_vf_features_extractor.parameters(), self.ego_value_net.parameters()),
            self.value_optimizer.param_groups[0]["lr"],
            **self.optimizer_kwargs,
        )

    def ego_value_forward(self, obs) -> th.Tensor:
        new_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        vf_features = self.ego_vf_features_extractor(new_obs)
        latent_vf = self.mlp_extractor.forward_ego_value(vf_features)
        return self.ego_value_net(latent_vf)

    def evaluate_ego_values(self, obs) -> th.Tensor:
        return self.ego_value_forward(obs)
