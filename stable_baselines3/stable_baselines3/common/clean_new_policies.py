import torch
import torch as th
import torch.autograd as autograd
import sys
import time
import random
from functools import partial
from venv import create
import wandb
import itertools
import numpy as np
import torch.nn as nn
from anyio import value
from gym import spaces
import warnings
from stable_baselines3.common.preprocessing import preprocess_obs
from .policies import BasePolicy, SelectLastLSTMOutput
from .distributions import BernoulliDistribution, CategoricalDistribution, DiagGaussianDistribution, MultiCategoricalDistribution, StateDependentNoiseDistribution, make_proba_distribution
from .preprocessing import maybe_transpose
from .type_aliases import GymEnv, MaybeCallback, Schedule
from .utils import obs_as_tensor, safe_mean, explained_variance, get_schedule_fn, \
    update_learning_rate, is_vectorized_observation
from .save_util import load_from_zip_file, recursive_getattr, recursive_setattr, \
    save_to_zip_file
from .vec_env import VecEnv
from .distributions import Distribution

from .buffers import DictRolloutBuffer, RolloutBuffer, ReplayBuffer, AdvRolloutBuffer
from .callbacks import BaseCallback
from .noise import ActionNoise
from .policies import ActorCriticPolicy, ActorCriticCnnPolicy, MultiInputActorCriticPolicy
from typing import Union, Type, Optional, Dict, Any, List, Tuple
#from stable_baselines3.common.clean_new_policies import CleanActorActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, MlpExtractorAdv, NatureCNN
from utils import select_matchup_env, move_optimizer_to_device

class PopArtHead(nn.Module):
    """Value head with output-preserving adaptive target normalization (PopArt).

    van Hasselt et al. 2016, "Learning values across many orders of magnitude".

    WHY HERE: the value targets on this task are tiny (logged value_loss is
    1.8e-5 - 2.2e-5, so RMS error ~0.004) AND non-stationary -- the return
    distribution's scale moves as the self-play policies move. Without
    normalization the head has to re-learn that scale by gradient descent,
    chasing a target that shifts underneath it. PopArt absorbs the scale
    analytically into (mu, sigma) instead.

    NOT what this fixes: the measured over-dispersion (affine slope 0.645-0.768,
    std V / std G ~0.65-0.70 against corr ~0.4-0.5). That is a bias/variance
    problem; normalizing targets does not change the V-vs-G correlation, so the
    slope will NOT move. Weight decay is the lever for that one.

    Also note Adam is (up to eps) invariant to a global rescale of the loss, so
    the usual "tiny targets -> tiny gradients" argument does NOT apply here. The
    benefit is the non-stationarity, not the magnitude.

    CONTRACT: `forward` returns DENORMALIZED values, i.e. real return units, the
    same as the bare Sequential it replaces. Every read site (evaluate_states,
    value_forward, LBR, the diagnostics) is therefore unchanged. Only the loss
    normalizes, and because
        (values_pred - mu) / sigma == net(x)      exactly
    it can do so from the denormalized prediction without needing a separate
    forward pass.
    """

    def __init__(self, net: nn.Module, beta: float = 3e-4,
                 sigma_min: float = 1e-4, sigma_max: float = 1e6):
        super().__init__()
        # Named `net` deliberately: state_dict keys become value_net.<key>.net.*
        # instead of value_net.<key>.*, which is why the caller must only build
        # this when --popart is on. Every existing .task predates it.
        self.net = net
        self.beta = float(beta)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        # Buffers, not plain attributes, so they ride along in state_dict and
        # survive checkpoint save/load without any bespoke serialization.
        self.register_buffer("mu", th.zeros(1))
        # nu MUST start at zero. The debias below is the standard Adam-style
        # correction nu/(1-(1-beta)^n), which assumes a ZERO-initialized EMA.
        # Initializing it to one instead leaves (1-beta)^n * 1.0 sitting in the
        # accumulator, and at beta=3e-4 that residual dominates for ~30k updates
        # (~3M env steps here) -- sigma read 0.694 against a true target scale of
        # 0.017, a 41x over-estimate, so the normalization did almost nothing.
        self.register_buffer("nu", th.zeros(1))         # running E[y^2]
        self.register_buffer("sigma", th.ones(1))       # pre-first-update value
        self.register_buffer("debias", th.zeros(1))     # 1-(1-beta)^n

    def forward(self, x):
        return self.net(x) * self.sigma + self.mu

    def normalize(self, y):
        """Real units -> normalized units. mu/sigma are detached by construction
        (buffers carry no grad), so this never backprops into the statistics."""
        return (y - self.mu) / self.sigma

    @th.no_grad()
    def update_stats(self, targets):
        """EMA-update (mu, sigma) on `targets`, then algebraically correct the
        final Linear so the module's OUTPUTS are unchanged by the rescale.

        Without the correction every statistics update would step-change every
        prediction the network makes -- which the policy is bootstrapping off.
        With it:
            sigma'(w'h + b') + mu' = sigma(wh + b) + mu      for all h
        """
        t = targets.detach().reshape(-1).float()
        if t.numel() == 0 or not th.isfinite(t).all():
            return
        mu_old, sigma_old = self.mu.clone(), self.sigma.clone()

        self.debias.mul_(1 - self.beta).add_(self.beta)
        self.mu.mul_(1 - self.beta).add_(self.beta * t.mean())
        self.nu.mul_(1 - self.beta).add_(self.beta * (t * t).mean())
        d = self.debias.clamp_min(1e-8)
        var = (self.nu / d) - (self.mu / d) ** 2
        self.sigma.copy_(var.clamp_min(self.sigma_min ** 2).sqrt()
                         .clamp(self.sigma_min, self.sigma_max))

        last = self.net[-1]
        if isinstance(last, nn.Linear):
            ratio = (sigma_old / self.sigma)
            last.weight.mul_(ratio)
            last.bias.mul_(ratio).add_((mu_old - self.mu) / self.sigma)
        # NOTE: we deliberately do NOT rescale the Adam moment estimates for
        # last.{weight,bias} here. They were accumulated under the pre-correction
        # parameterization and are now stale; empirically they re-adapt within a
        # few hundred steps, and rescaling them is not part of the original
        # PopArt formulation. May revisit -- if the value loss shows a
        # transient spike right after each stats update, this is the first
        # suspect.

    def effective_stats(self):
        return float(self.mu.item()), float(self.sigma.item())


class MinimaxHead(nn.Module):
    """Joint-action critic: emits the whole (n_ego x n_adv) payoff matrix per state.

    Q(s, a_ego, a_adv), ego payoff, zero-sum -- so the adversary's value is just
    -Q and the six ego/adv sign-negation sites collapse to one object.

    WHY A MATRIX AND NOT AN ACTION-CONDITIONED SCALAR: the inner minimax solve
    needs ALL n*m entries at s' to compute V(s'). An action-conditioned head
    (q_ppo's shape: latent + actions -> scalar) would need n*m = 484 forwards
    per state, ~500k per 1024-batch update. One forward emitting 484 outputs is
    the same trunk and 248k extra params.

    WHY THIS OBJECT AT ALL: every critic intervention tried through 2026-08-08
    (gamma, V-trace, replay capacity, PopArt) changed how well V predicts
    returns across the state distribution, and none created sensitivity to
    single-action differences. `lbr ~= shuffle` over 42 measurements
    (-0.1419 vs -0.1407) says permuting V across the 88 LBR branches changes
    nothing: V(s') is constant across branches until the states diverge.
    Q(s,a,o) varies across `a` by construction. That is the entire bet, and it
    is a bet -- see the gate in the plan before wiring this into training.

    ONLY ONE ENTRY PER TRANSITION GETS A GRADIENT (the joint action actually
    played); the other 483 are trained by other visits to the same state. So Q
    is far hungrier than V, and V is already at its supervised ceiling. Watch
    `coverage()` -- if most cells never receive a gradient, this cannot work.
    """

    def __init__(self, trunk: nn.Module, latent_dim: int, n_ego: int, n_adv: int):
        super().__init__()
        self.n_ego = int(n_ego)
        self.n_adv = int(n_adv)
        self.trunk = trunk
        self.out = nn.Linear(latent_dim, self.n_ego * self.n_adv)
        # Cell-visit counts, for the coverage diagnostic above. A buffer so it
        # rides along in state_dict and survives checkpointing.
        self.register_buffer("cell_visits", th.zeros(self.n_ego, self.n_adv))

    def forward(self, latent_vf: th.Tensor) -> th.Tensor:
        """(B, latent) -> (B, n_ego, n_adv) payoff matrices."""
        h = self.trunk(latent_vf)
        return self.out(h).view(-1, self.n_ego, self.n_adv)

    def played(self, latent_vf: th.Tensor, a_ego: th.Tensor, a_adv: th.Tensor) -> th.Tensor:
        """Q at the joint action actually taken. (B,) -- this is what the loss regresses."""
        M = self(latent_vf)
        b = th.arange(M.shape[0], device=M.device)
        return M[b, a_ego.long().reshape(-1), a_adv.long().reshape(-1)]

    @th.no_grad()
    def note_visits(self, a_ego: th.Tensor, a_adv: th.Tensor) -> None:
        idx = (a_ego.long().reshape(-1) * self.n_adv + a_adv.long().reshape(-1))
        flat = self.cell_visits.view(-1)
        flat.scatter_add_(0, idx, th.ones_like(idx, dtype=flat.dtype))

    def coverage(self) -> float:
        """Fraction of the n_ego*n_adv cells that have ever received a gradient."""
        return float((self.cell_visits > 0).float().mean())


class CleanActorActorCriticPolicy(ActorCriticPolicy):
    def __init__(self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        # TODO(antonin): update type annotation when we remove shared network support
        net_arch: Union[List[int], Dict[str, List[int]], List[Dict[str, List[int]]], None] = None,
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = False,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        matchups=None,
        envs_per_matchup=None,
        num_adversaries=None,
        dstb_action_space=None,
        use_mirror: bool = False,
        side_dim: int = 1,
        # PopArt target normalization on the value heads. OFF by default and it
        # must stay that way: turning it on wraps each value_net head, which
        # renames state_dict keys (value_net.<k>.* -> value_net.<k>.net.*) and
        # would break loading every checkpoint that predates it.
        popart: bool = False,
        popart_beta: float = 3e-4,
        # Minimax-Q joint-action critic. OFF by default: turning it on changes
        # q_value_net's output from a scalar to an (n_ego x n_adv) matrix and
        # renames state_dict keys, so no earlier checkpoint loads into it.
        minimax_q: bool = False,
        minimax_n_ego: int = 22,
        minimax_n_adv: int = 22,
    ):
        # self.matchups = matchups
        # self.envs_per_matchup = envs_per_matchup
        # self.num_adversaries = num_adversaries
        # self.dstb_action_space = dstb_action_space
        # self.use_sde = use_sde
        # self.dist_kwargs = None
        # self.f
        # if dstb_action_space is None:
        #     self.dstb_action_space = action_space
        #     dstb_action_space = action_space
        # self.dstb_action_dist = [make_proba_distribution(self.dstb_action_space, use_sde=self.use_sde, dist_kwargs=None) for i in range(self.num_adversaries)]
        # self.pi_dstb_features_extractor = self.make_features_extractor()
        self.use_mirror = use_mirror
        self.side_dim = side_dim
        self.popart = bool(popart)
        self.popart_beta = float(popart_beta)
        self.minimax_q = bool(minimax_q)
        self.minimax_n_ego = int(minimax_n_ego)
        self.minimax_n_adv = int(minimax_n_adv)
        super().__init__(observation_space = observation_space,
        action_space = action_space,
        lr_schedule = lr_schedule[0],
        # TODO(antonin): update type annotation when we remove shared network support
        net_arch = net_arch,
        activation_fn = activation_fn,
        ortho_init = ortho_init,
        use_sde = use_sde,
        log_std_init = log_std_init,
        full_std = full_std,
        use_expln = use_expln,
        squash_output = squash_output,
        features_extractor_class = features_extractor_class,
        features_extractor_kwargs = features_extractor_kwargs,
        share_features_extractor = share_features_extractor,
        normalize_images = normalize_images,
        optimizer_class = optimizer_class,
        optimizer_kwargs = optimizer_kwargs,
    )
        self.dstb_action_space = dstb_action_space
        if dstb_action_space is None:
            self.dstb_action_space = action_space
            dstb_action_space = action_space
        # self.use_sde = use_sde
        # self.dist_kwargs = None
        self.num_adversaries = num_adversaries
        self.matchups = matchups
        self.envs_per_matchup = envs_per_matchup
        self.dstb_action_dist = [make_proba_distribution(self.dstb_action_space, use_sde=self.use_sde, dist_kwargs=None) for i in range(self.num_adversaries)]
        self.pi_dstb_features_extractor = self.make_features_extractor()
        self.pi_ctrl_features_extractor = self.features_extractor
        #self.vf_features
        net_arch = dict(pi=[256,256], vf=[512,512])
        self.net_arch = net_arch
        self._build_network(lr_schedule)
        print("hello")

    def _build_mlp_extractor(self, extra=False) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractorAdv(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device='auto',
            adversarial=True,
            context_dim=0,
            side_dim=self.side_dim if self.use_mirror else 0,
        )

    def _build_network(self, joint_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        # Actor-head width. Kept at 256 so widening the critic does not silently
        # change the policy as well -- otherwise an improvement can't be attributed.
        lstm_hidden_size = 256
        # Critic-head width, split out from lstm_hidden_size so the value path can
        # be widened independently. Measured motivation: the 512->256 critic trunk
        # (mlp_extractor.value_net) discards ~0.2 EV of return-predictive signal,
        # while a RANDOM 512->256 projection of the same shape discards none --
        # i.e. training makes that module worse than an untrained one.
        value_hidden_size = 512
        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
            self.dstb_action_net = nn.ModuleDict()
            self.dstb_log_std = {}  # Store log_std Parameters in a regular dict since they're not Modules
            for i in range(self.num_adversaries):
                key = select_matchup_env(self.matchups, i, self.envs_per_matchup)
                mean_net, log_std_param = self.dstb_action_dist[i].proba_distribution_net(latent_dim=latent_dim_pi, log_std_init=self.log_std_init)
                self.dstb_action_net[key] = mean_net  # Store only the Module in ModuleDict
                self.dstb_log_std[key] = log_std_param  # Store Parameter in regular dict
            #self.dstb_action_net, self.dstb_log_std = self.dstb_action_dist[0].proba_distribution_net(latent_dim=latent_dim_pi, log_std_init=self.log_std_init)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
            if self.adversarial: # we are doing adversarial!
                self.dstb_action_net, self.dstb_log_std = self.dstb_action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            #self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)

            self.action_net = nn.Sequential(
                #nn.LSTM(input_size=latent_dim_pi, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True),
                #SelectLastLSTMOutput(),
                nn.Linear(latent_dim_pi, lstm_hidden_size),
                self.activation_fn(),
                nn.Linear(lstm_hidden_size, latent_dim_pi),
                self.activation_fn(),
                self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
            )

            self.dstb_action_net = nn.ModuleDict()
            self.head_length = 10 # lstm = 4, 2 linear layers = 2 + 2, proba_dist is also a linear, so 2, total 10
            for i in range(self.num_adversaries):
                matchup_key = select_matchup_env(self.matchups, i, self.envs_per_matchup)
                self.dstb_action_net[matchup_key] = nn.Sequential(
                    #nn.LSTM(input_size=latent_dim_pi, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True),
                    #SelectLastLSTMOutput(),
                    nn.Linear(latent_dim_pi, lstm_hidden_size),
                    self.activation_fn(),
                    nn.Linear(lstm_hidden_size, latent_dim_pi),
                    self.activation_fn(),
                    self.dstb_action_dist[i].proba_distribution_net(latent_dim=latent_dim_pi))
                    
                #if i == 0:
                #    assert len(next(iter(self.dstb_action_net.values()))) == 7 and self.head_length == 10

                #self.dstb_action_net.append(self.dstb_action_dist[i].proba_distribution_net(latent_dim=latent_dim_pi))
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")
        
        self.value_net = nn.ModuleDict()
        self.q_value_net = nn.ModuleDict()
        self.minimax_net = nn.ModuleDict()   # populated only when minimax_q
        for i in range(self.num_adversaries):
            matchup_key = select_matchup_env(self.matchups, i, self.envs_per_matchup)
            # LSTM removed. It was never trained (q_value_net was bit-identical
            # across 9.1M steps), it makes checkpoints unloadable by
            # critic_diagnostics.py -- which is why the whole q head has been
            # undiagnosable -- and recurrence is the wrong tool here: minimax-Q
            # wants a per-state payoff MATRIX, not a sequence summary. Now
            # feed-forward and shaped like value_net so the two are comparable
            # tap-for-tap in the diagnostics.
            # NOTE: the minimax matrix head does NOT live here. This trunk is fed
            # [vf_features || ego_action_emb || adv_action_emb] by
            # mlp_extractor.forward_q_value, so its input dim assumes actions are
            # concatenated -- a state-only matrix forward cannot reuse it. And
            # q_value_optimizer includes vf_features_extractor, which is shared
            # with value_optimizer, so stepping it would apply AdamW's decoupled
            # weight decay to the shared CNN a SECOND time per update even under
            # stop-grad. The matrix head lives in self.minimax_net below, off
            # forward_critic, with an optimizer scoped to itself.
            self.q_value_net[matchup_key] = nn.Sequential(
                nn.Linear(self.mlp_extractor.latent_dim_vf, value_hidden_size),
                self.activation_fn(),
                nn.Linear(value_hidden_size, value_hidden_size),
                self.activation_fn(),
                nn.Linear(value_hidden_size, value_hidden_size),
                self.activation_fn(),
                nn.Linear(value_hidden_size, 1))
            _vhead = nn.Sequential(
                nn.Linear(self.mlp_extractor.latent_dim_vf, value_hidden_size),
                self.activation_fn(),
                nn.Linear(value_hidden_size, value_hidden_size),
                self.activation_fn(),
                nn.Linear(value_hidden_size, value_hidden_size),
                self.activation_fn(),
                nn.Linear(value_hidden_size, 1))
            # Only wrap when enabled -- an unwrapped head keeps the historical
            # state_dict layout, so every existing .task still loads.
            self.value_net[matchup_key] = (
                PopArtHead(_vhead, beta=getattr(self, "popart_beta", 3e-4))
                if getattr(self, "popart", False) else _vhead)
            #self.value_net.append(nn.Linear(self.mlp_extractor.latent_dim_vf, 1))

            # Minimax-Q joint-action critic. Its own ModuleDict off
            # forward_critic (the same latent value_net consumes), NOT hung on
            # q_value_net -- see the note there. Built only when enabled, so
            # with the flag off the module tree and state_dict are unchanged.
            if getattr(self, "minimax_q", False):
                self.minimax_net[matchup_key] = MinimaxHead(
                    nn.Sequential(
                        nn.Linear(self.mlp_extractor.latent_dim_vf, value_hidden_size),
                        self.activation_fn(),
                        nn.Linear(value_hidden_size, value_hidden_size),
                        self.activation_fn(),
                        nn.Linear(value_hidden_size, value_hidden_size),
                        self.activation_fn()),
                    value_hidden_size,
                    n_ego=self.minimax_n_ego, n_adv=self.minimax_n_adv)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.dstb_action_net: 0.01,
                self.value_net: 1,
            }
            
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        
        _film = lambda attr: getattr(self.mlp_extractor, attr).parameters() if hasattr(self.mlp_extractor, attr) else iter([])

        if len(self.mlp_extractor.policy_net) == 0:
            self.ctrl_optimizer = self.optimizer_class(
                itertools.chain(self.pi_ctrl_features_extractor.parameters(), self.action_net.parameters(), _film('policy_film')),
                joint_schedule[1](1), maximize=False)
            self.dstb_optimizer = self.optimizer_class(
                itertools.chain(self.pi_dstb_features_extractor.parameters(), self.dstb_action_net.parameters(), _film('dstb_film')),
                joint_schedule[2](1), maximize=False)
            self.value_optimizer = self.optimizer_class(
                itertools.chain(self.vf_features_extractor.parameters(), self.value_net.parameters(), _film('value_film'), _film('ego_value_film')),
                joint_schedule[0](1), **self.optimizer_kwargs)
        else:
            self.ctrl_optimizer = self.optimizer_class(itertools.chain(self.mlp_extractor.policy_net.parameters(), self.pi_ctrl_features_extractor.parameters(), self.action_net.parameters(), _film('policy_film')), joint_schedule[0](1), maximize=False)
            if isinstance(self.action_dist, DiagGaussianDistribution):
                # Collect all log_std parameters for all adversaries - need to wrap in iterables for chain
                log_std_params = [self.dstb_log_std[select_matchup_env(self.matchups, i, self.envs_per_matchup)] for i in range(self.num_adversaries)]
                self.dstb_optimizer = self.optimizer_class(itertools.chain(self.mlp_extractor.dstb_net.parameters(), self.pi_dstb_features_extractor.parameters(), self.dstb_action_net.parameters(), iter(log_std_params), _film('dstb_film')), joint_schedule[1](1), maximize=False)
            else:
                self.dstb_optimizer = self.optimizer_class(itertools.chain(self.mlp_extractor.dstb_net.parameters(), self.pi_dstb_features_extractor.parameters(), self.dstb_action_net.parameters(), _film('dstb_film')), joint_schedule[1](1), maximize=False)
            self.extractor_and_trunk_length = 12
            #self.value_optimizer = self.optimizer_class(
            #    itertools.chain(self.mlp_extractor.value_net.parameters(), self.vf_features_extractor.parameters(), itertools.chain.from_iterable([self.value_net[i].parameters() for i in range(self.num_adversaries)])),
            #    joint_schedule[2](1), **self.optimizer_kwargs)
            # with th.no_grad():
            #     for param in itertools.chain(
            #         self.mlp_extractor.value_net.parameters(),
            #         self.vf_features_extractor.parameters(),
            #         self.value_net.parameters(),
            #     ):
            #         param.fill_(999.0)
            self.value_optimizer = self.optimizer_class(
                itertools.chain(self.mlp_extractor.value_net.parameters(), self.vf_features_extractor.parameters(), self.value_net.parameters(), _film('value_film'), _film('ego_value_film')),
                joint_schedule[2](1), **self.optimizer_kwargs)
            #self.value_targ = [copy.deepcopy(self.vf_features_extractor).requires_grad_(False).to('cuda'),
            #                   copy.deepcopy(self.mlp_extractor.value_net).requires_grad_(False).to('cuda'),
            #                   [copy.deepcopy(self.value_net)[i].requires_grad_(False).to('cuda') for i in range(len(self.value_net))]]
            self.q_value_optimizer = self.optimizer_class(
                itertools.chain(self.mlp_extractor.q_value_net.parameters(), self.vf_features_extractor.parameters(), self.q_value_net.parameters(), self.mlp_extractor.ego_action_extractor.parameters(), self.mlp_extractor.adv_action_extractor.parameters(), _film('q_value_film')),
                joint_schedule[2](1), **self.optimizer_kwargs)

            # Minimax optimizer: ONLY the matrix heads' own parameters.
            #
            # Deliberately excludes vf_features_extractor even though the head
            # consumes its output, because two AdamW instances sharing a
            # parameter is a live hazard. Measured, with one param in both:
            #     grad = None    -> both inert (AdamW skips None grads)
            #     grad = zeros   -> BOTH apply weight decay      (1.4e-06 each)
            #     grad = real    -> BOTH apply a full Adam step  (4.0e-04 each)
            # So it is safe only while the grad is None at the moment the second
            # optimizer steps -- i.e. it depends on zero_grad(set_to_none=) and
            # on step ordering, both easy to change without noticing. Scoping
            # this optimizer to its own parameters removes the dependence
            # entirely, so --minimax_q True stays identical to False for
            # everything except the new head regardless of call order.
            #
            # NOTE the same shared-parameter pattern already exists between
            # value_optimizer and q_value_optimizer (both hold
            # vf_features_extractor). Latent today because the q path is unused;
            # it would bite if that head is ever trained in the same loop.
            if getattr(self, "minimax_q", False):
                self.minimax_optimizer = self.optimizer_class(
                    self.minimax_net.parameters(),
                    joint_schedule[2](1), **self.optimizer_kwargs)

    def _get_ego_action_dist_from_latent(self, latent_pi) -> Tuple[Distribution, Distribution]:
        mean_actions = self.action_net(latent_pi)
        
        if isinstance(self.action_dist, BernoulliDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        elif isinstance(self.action_dist, CategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        raise ValueError("Invalid action distribution")

    # def _get_adv_action_dist_from_latent(self, latent_pi_dstb, buf_num, evaluate=False) -> Tuple[Distribution, Distribution]:
    #     if evaluate:
    #         assert len(buf_num) == 1
    #     dstb_actions = th.zeros((latent_pi_dstb.shape[0], self.dstb_action_space.shape[0])).to(self.device)
    #     latents_per_adv = latent_pi_dstb.shape[0] // self.num_adversaries
    #     for i in range(len(buf_num)):
    #         key = select_matchup_env(self.matchups, buf_num[i], self.envs_per_matchup)
    #         # Check if distribution is DiagGaussian (has mean/log_std structure) vs Bernoulli/Categorical
    #         if isinstance(self.dstb_action_dist[buf_num[i]], DiagGaussianDistribution):
    #             # For DiagGaussian, dstb_action_net contains the mean network directly
    #             dstb_action_net_to_use = self.dstb_action_net[key]
    #         else:
    #             dstb_action_net_to_use = self.dstb_action_net[key]
    #         if evaluate:
    #             dstb_actions = dstb_action_net_to_use(latent_pi_dstb)
    #             return self.dstb_action_dist[buf_num[0]].proba_distribution(action_logits=dstb_actions)
    #         else:
    #             dstb_actions[buf_num[i] * latents_per_adv : (buf_num[i]+1) * latents_per_adv, :] = dstb_action_net_to_use(latent_pi_dstb[buf_num[i] * latents_per_adv : (buf_num[i]+1) * latents_per_adv, :])
    #     if isinstance(self.dstb_action_dist[buf_num[i]], BernoulliDistribution):
    #         return [self.dstb_action_dist[buf_num[i]].proba_distribution(action_logits=dstb_actions[buf_num[i] * latents_per_adv : (buf_num[i]+1) * latents_per_adv, :]) for i in range(len(buf_num))] 
    #     elif isinstance(self.dstb_action_dist[buf_num[i]], DiagGaussianDistribution):
    #         return [self.dstb_action_dist[buf_num[i]].proba_distribution(mean_actions=dstb_actions[buf_num[i] * latents_per_adv : (buf_num[i]+1) * latents_per_adv, :], log_std=self.dstb_log_std[select_matchup_env(self.matchups, buf_num[i], self.envs_per_matchup)]) for i in range(len(buf_num))]
    #     else:
    #         raise ValueError("Invalid action distribution")
    def _get_adv_action_dist_from_latent(self, latent_pi_dstb, buf_num, evaluate=False) -> Tuple[Distribution, Distribution]:
        if evaluate or len(buf_num) == 1:
            assert len(buf_num) == 1
            num_adversaries = 1
            evaluate = True
        else:
            num_adversaries = self.num_adversaries
        sample_dstb_net = next(iter(self.dstb_action_net.values()))
        head = sample_dstb_net[-1] if isinstance(sample_dstb_net, nn.Sequential) else sample_dstb_net
        n_logits = head.out_features
        mean_or_logit_dstb_actions = th.zeros((latent_pi_dstb.shape[0], n_logits), device=self.device)
        latents_per_adv = latent_pi_dstb.shape[0] // num_adversaries
        for i in range(len(buf_num)):
            chunk = slice(buf_num[i] * latents_per_adv, (buf_num[i]+1) * latents_per_adv)
            if evaluate:
                chunk = slice(0 * latents_per_adv, 1 * latents_per_adv)
            key = select_matchup_env(self.matchups, buf_num[i], self.envs_per_matchup)
            # Check if distribution is DiagGaussian (has mean/log_std structure) vs Bernoulli/Categorical
            dstb_action_net_to_use = self.dstb_action_net[key]
            mean_or_logit_dstb_actions[chunk] = dstb_action_net_to_use(latent_pi_dstb[chunk])
            # if evaluate:
            #     dstb_actions = dstb_action_net_to_use(latent_pi_dstb)
            #     if isinstance(self.dstb_action_dist[buf_num[0]], DiagGaussianDistribution):
            #         key = select_matchup_env(self.matchups, buf_num[0], self.envs_per_matchup)
            #         return self.dstb_action_dist[buf_num[0]].proba_distribution(mean_actions=dstb_actions, log_std=self.dstb_log_std[key])
            #     else:
            #         return self.dstb_action_dist[buf_num[0]].proba_distribution(action_logits=dstb_actions)
            # else:
            #     dstb_actions[buf_num[i] * latents_per_adv : (buf_num[i]+1) * latents_per_adv, :] = dstb_action_net_to_use(latent_pi_dstb[buf_num[i] * latents_per_adv : (buf_num[i]+1) * latents_per_adv, :])
        distributions = []
        if isinstance(self.dstb_action_dist[buf_num[i]], BernoulliDistribution):
            for i in range(len(buf_num)):
                chunk = slice(buf_num[i] * latents_per_adv, (buf_num[i]+1) * latents_per_adv)
                if evaluate:
                    chunk = slice(0 * latents_per_adv, 1 * latents_per_adv)
                distributions.append(self.dstb_action_dist[buf_num[i]].proba_distribution(action_logits=mean_or_logit_dstb_actions[chunk]))
            return distributions
            #return [self.dstb_action_dist[buf_num[i]].proba_distribution(action_logits=dstb_actions[buf_num[i] * latents_per_adv : (buf_num[i]+1) * latents_per_adv, :]) for i in range(len(buf_num))] 
        elif isinstance(self.dstb_action_dist[buf_num[i]], DiagGaussianDistribution):

            for i in range(len(buf_num)):
                key = select_matchup_env(self.matchups, buf_num[i], self.envs_per_matchup)
                chunk = slice(buf_num[i] * latents_per_adv, (buf_num[i]+1) * latents_per_adv)
                if evaluate:
                    chunk = slice(0 * latents_per_adv, 1 * latents_per_adv)
                distributions.append(self.dstb_action_dist[buf_num[i]].proba_distribution(mean_actions=mean_or_logit_dstb_actions[chunk], log_std=self.dstb_log_std[key]))
            return distributions
            #return [self.dstb_action_dist[buf_num[i]].proba_distribution(mean_actions=dstb_actions[buf_num[i] * latents_per_adv : (buf_num[i]+1) * latents_per_adv, :], log_std=self.dstb_log_std[select_matchup_env(self.matchups, buf_num[i], self.envs_per_matchup)]) for i in range(len(buf_num))]
        elif isinstance(self.dstb_action_dist[buf_num[i]], MultiCategoricalDistribution):
            for i in range(len(buf_num)):
                key = select_matchup_env(self.matchups, buf_num[i], self.envs_per_matchup)
                chunk = slice(buf_num[i] * latents_per_adv, (buf_num[i]+1) * latents_per_adv)
                if evaluate:
                    chunk = slice(0 * latents_per_adv, 1 * latents_per_adv)
                distributions.append(self.dstb_action_dist[buf_num[i]].proba_distribution(action_logits=mean_or_logit_dstb_actions[chunk]))
            return distributions
        else:
            raise ValueError("Invalid action distribution")
        
    def ego_forward(self, obs, deterministic=False, side_flag=None) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        new_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        pi_ctrl_features = self.pi_ctrl_features_extractor(new_obs)
        if self.use_mirror:
            latent_pi = self.mlp_extractor.ego_forward(pi_ctrl_features, side_flag=th.ones(pi_ctrl_features.shape[0], 1).to(self.device) * side_flag)
        else:
            latent_pi = self.mlp_extractor.ego_forward(pi_ctrl_features)
        ctrl_distribution = self._get_ego_action_dist_from_latent(latent_pi)
        ctrl_actions = ctrl_distribution.get_actions(deterministic=deterministic)
        ctrl_log_prob = ctrl_distribution.log_prob(ctrl_actions)
        return ctrl_actions, ctrl_log_prob

    def adv_forward(self, obs, buf_num=None, deterministic=False, side_flag=None) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        if buf_num is None:
            buf_num = [i for i in range(self.num_adversaries)]
        new_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        pi_dstb_features = self.pi_dstb_features_extractor(new_obs)
        latent_pi_dstb = self.mlp_extractor.adv_forward(pi_dstb_features, side_flag=side_flag)
        dstb_distribution = self._get_adv_action_dist_from_latent(latent_pi_dstb, buf_num=buf_num)
        dstb_actions = [dstb_distribution[i].get_actions(deterministic=deterministic) for i in range(len(dstb_distribution))]
        #dstb_actions = [dstb_actions[i].reshape((-1, *self.dstb_action_space.shape)) for i in range(self.num_adversaries)]
        #dstb_actions = th.vstack(dstb_actions)
        dstb_log_prob = [dstb_distribution[i].log_prob(dstb_actions[i]) for i in range(len(dstb_distribution))]
        #dstb_actions = th.vstack(dstb_actions)
        #test = th.zeros((dstb_actions.shape[0],))
        #for i in range(self.num_adversaries):
        #    test[i * (self.envs_per_matchup): (i + 1) * self.envs_per_matchup] = dstb_log_prob[i][:]
        dstb_actions = th.vstack(dstb_actions)
        dstb_log_prob = th.hstack(dstb_log_prob)
        #dstb_log_prob = test
        return dstb_actions, dstb_log_prob

    def minimax_matrices(self, obs, buf_num=None, side_flag=None, stop_grad=True):
        """Q(s, ., .) payoff matrices for a batch of observations. (B, n_ego, n_adv)

        stop_grad=True (PHASE 0) detaches the shared latent, so the matrix head
        trains without its gradients ever reaching vf_features_extractor or
        mlp_extractor.value_net. Together with minimax_optimizer being scoped to
        the head's own parameters, that makes enabling this head a no-op for the
        policy -- which is what lets the Phase 0 gate be a clean measurement
        rather than a confounded one.
        """
        if not getattr(self, "minimax_q", False):
            raise RuntimeError("minimax_matrices called but --minimax_q is off")
        def _shared():
            new_obs = preprocess_obs(obs, self.observation_space,
                                     normalize_images=self.normalize_images)
            vf_features = self.vf_features_extractor(new_obs)
            if self.use_mirror and side_flag is not None:
                return self.mlp_extractor.forward_critic(
                    vf_features,
                    side_flag=th.ones(vf_features.shape[0], 1).to(self.device) * side_flag)
            return self.mlp_extractor.forward_critic(vf_features)

        if stop_grad:
            # no_grad, not just .detach(): detaching afterwards still records the
            # CNN forward on the tape, so we would pay to build a graph that is
            # then thrown away. The encoder is the expensive part of this pass.
            with th.no_grad():
                latent_vf = _shared()
        else:
            latent_vf = _shared()
        # Single-matchup fast path; the loop below is the general case.
        if buf_num is not None and len(buf_num) == 1:
            key = select_matchup_env(self.matchups, buf_num[0], self.envs_per_matchup)
            return self.minimax_net[key](latent_vf)
        per = latent_vf.shape[0] // self.num_adversaries
        out = []
        for i in range(self.num_adversaries):
            key = select_matchup_env(self.matchups, i, self.envs_per_matchup)
            out.append(self.minimax_net[key](latent_vf[i * per:(i + 1) * per]))
        return th.cat(out, dim=0)

    def minimax_head_for(self, buf_num):
        """The MinimaxHead for this update, or None when minimax_q is off."""
        if not getattr(self, "minimax_q", False):
            return None
        if buf_num is None or len(buf_num) != 1:
            return None
        key = select_matchup_env(self.matchups, buf_num[0], self.envs_per_matchup)
        # nn.ModuleDict is not a dict -- no .get(). It supports `in` and [].
        return self.minimax_net[key] if key in self.minimax_net else None

    def popart_for(self, buf_num):
        """The PopArtHead for this update, or None when PopArt is off.

        Returns None (i.e. fall back to the plain loss) when the minibatch spans
        more than one matchup head, because a single (mu, sigma) cannot describe
        several heads at once. Guessing there would silently normalize one head's
        targets by another head's scale.
        """
        if not getattr(self, "popart", False):
            return None
        if buf_num is None or len(buf_num) != 1:
            return None
        key = select_matchup_env(self.matchups, buf_num[0], self.envs_per_matchup)
        head = self.value_net[key]
        return head if isinstance(head, PopArtHead) else None

    def value_forward(self, obs, side_flag=None) -> Tuple[th.Tensor, th.Tensor]:
        new_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        vf_features = self.vf_features_extractor(new_obs)
        if self.use_mirror:
            latent_vf = self.mlp_extractor.forward_critic(vf_features, side_flag=th.ones(vf_features.shape[0], 1).to(self.device) * side_flag)
        else:
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        latents_per_adv = latent_vf.shape[0] // self.num_adversaries
        values = th.zeros((latent_vf.shape[0], 1), device=self.device)
        for i in range(self.num_adversaries):
            # need to test
            key = select_matchup_env(self.matchups, i, self.envs_per_matchup)
            values[i * latents_per_adv : (i+1) * latents_per_adv, :] = self.value_net[key](latent_vf[i * latents_per_adv : (i+1) * latents_per_adv, :])
        return values

    def q_value_forward(self, obs, ego_actions, adv_actions, side_flag=None) -> Tuple[th.Tensor, th.Tensor]:
        new_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        vf_features = self.vf_features_extractor(new_obs)
        if self.use_mirror:
            latent_vf = self.mlp_extractor.forward_q_value(vf_features, ego_actions, adv_actions, side_flag=side_flag)
        else:
            latent_vf = self.mlp_extractor.forward_q_value(vf_features, ego_actions, adv_actions)
        latents_per_adv = latent_vf.shape[0] // self.num_adversaries
        q_values = th.zeros((latent_vf.shape[0], 1), device=self.device)
        for i in range(self.num_adversaries):
            key = select_matchup_env(self.matchups, i, self.envs_per_matchup)
            q_values[i * latents_per_adv : (i+1) * latents_per_adv, :] = self.q_value_net[key](latent_vf[i * latents_per_adv : (i+1) * latents_per_adv, :])
        return q_values

    def forward(self, obs, deterministic=False, ego_forward=True, adv_forward=True, network_keys=None, zero_ego_action=False, zero_adv_action=False, random_ego_action=False, random_adv_action=False, value_forward=True, q_value_forward=True, ego_side_flag=None, adv_side_flag=None) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:

        # by default, we run both ego and adv forward

        if ego_forward:
            if self.use_mirror:
                if ego_side_flag is not None:
                    ego_actions, ego_log_prob = self.ego_forward(obs, deterministic, side_flag=ego_side_flag)
                else:
                    left_ego_actions, left_ego_log_prob = self.ego_forward(obs, deterministic, side_flag=th.Tensor([0]).to(self.device))
                    right_ego_actions, right_ego_log_prob = self.ego_forward(obs, deterministic, side_flag=th.Tensor([1]).to(self.device))
                    ego_actions = th.cat([left_ego_actions, right_ego_actions], dim=0)
                    ego_log_prob = th.cat([left_ego_log_prob, right_ego_log_prob], dim=0)
            else:
                ego_actions, ego_log_prob = self.ego_forward(obs, deterministic)
        if zero_ego_action:
            # Mirror the zero_adv_action branch below: clone the real action tensor
            # so dtype and batch shape are inherited. Constructing a fresh
            # th.ones(...) here gave float32 actions (real ones are int64 from the
            # MultiCategorical), which propagated through np.hstack and made
            # SFWrapper.action_transformer index DIRECTIONS_BUTTONS with a
            # numpy.float64. It also hardcoded the batch to
            # num_adversaries*envs_per_matchup (wrong under mirror, where the batch
            # is doubled) and used ones -- i.e. action 1 = ['UP'], a permanent jump
            # rather than the intended no-op.
            if ego_forward is False:
                raise ValueError("Cannot zero ego actions if ego forward is False")
            ego_actions = th.zeros_like(ego_actions)
            ego_log_prob = th.zeros_like(ego_log_prob)
        if random_ego_action:
            if ego_forward is False:
                raise ValueError("Cannot random ego actions if ego forward is False")
            if not hasattr(self, '_random_ego_step'):
                self._random_ego_step = 0
            if hasattr(self.action_space, 'nvec'):
                action_val = self._random_ego_step % int(self.action_space.nvec[0])
                ego_actions = th.full_like(ego_actions, action_val)
            else:
                action_val = self._random_ego_step % 2
                ego_actions = th.full_like(ego_actions, action_val)
            self._random_ego_step += 1
            ego_log_prob = th.zeros_like(ego_log_prob)
        if adv_forward:
            if self.use_mirror:
                if adv_side_flag is not None:
                    adv_actions, adv_log_prob = self.adv_forward(obs, deterministic=deterministic, side_flag=adv_side_flag)
                else:
                    n = obs.shape[0]
                    adv_side = th.cat([th.ones(n // 2, 1), th.zeros(n // 2, 1)], dim=0).to(self.device)
                    adv_actions, adv_log_prob = self.adv_forward(obs, deterministic=deterministic, side_flag=adv_side)
            else:
                adv_actions, adv_log_prob = self.adv_forward(obs, deterministic=deterministic)
            #adv_actions = adv_actions[0]
            #adv_log_prob = adv_log_prob[0]
        if zero_adv_action:
            if adv_forward is False:
                raise ValueError("Cannot zero adv actions if adv forward is False")
            adv_actions = th.zeros_like(adv_actions)
            adv_log_prob = th.zeros_like(adv_log_prob)
            #adv_entropy = th.zeros()
        if random_adv_action:
            if adv_forward is False:
                raise ValueError("Cannot random adv actions if adv forward is False")
            if not hasattr(self, '_random_adv_step'):
                self._random_adv_step = 1
            if hasattr(self.action_space, 'nvec'):
                action_val = self._random_adv_step % int(self.action_space.nvec[0])
                adv_actions = th.full_like(adv_actions, action_val)
            else:
                action_val = self._random_adv_step % 2
                adv_actions = th.full_like(adv_actions, action_val)
            self._random_adv_step += 1
            adv_log_prob = th.zeros_like(adv_log_prob)

        if value_forward:
            if self.use_mirror:
                # Mirror callers should pass value_forward=False and call
                # value_forward() directly with per-env side flags.
                raise ValueError("value_forward=True is not supported in mirror mode; call value_forward() directly with per-env side_flags")
            else:
                values = self.value_forward(obs)
        else:
            values = th.zeros(ego_actions.shape[0], 1) if ego_forward else th.zeros(adv_actions.shape[0], 1)
        #q_values = self.q_value_forward(obs, ego_actions, adv_actions)
        return ego_actions, ego_log_prob, adv_actions, adv_log_prob, values, th.zeros_like(values)

    def evaluate_ego_actions(self, obs, ego_actions, side_flag=None) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        features = self.pi_ctrl_features_extractor(preprocessed_obs)
        if self.use_mirror:
            latent_pi = self.mlp_extractor.ego_forward(features, side_flag=side_flag)
        else:
            latent_pi = self.mlp_extractor.ego_forward(features)
        ctrl_distribution = self._get_ego_action_dist_from_latent(latent_pi)
        ctrl_log_prob = ctrl_distribution.log_prob(ego_actions)
        ctrl_entropy = ctrl_distribution.entropy()
        return ctrl_log_prob, ctrl_entropy
    
    def evaluate_adv_actions(self, obs, adv_actions, buf_num, side_flag=None) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        assert len(buf_num) == 1
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        features = self.pi_dstb_features_extractor(preprocessed_obs)
        actions_per_adv = adv_actions.shape[0] // self.num_adversaries
        latent_pi_dstb = self.mlp_extractor.adv_forward(features, side_flag=side_flag)
        dstb_distribution = self._get_adv_action_dist_from_latent(latent_pi_dstb, buf_num, evaluate=True)
        #dstb_log_prob = dstb_distribution.log_prob(adv_actions)
        dstb_log_prob = dstb_distribution[0].log_prob(adv_actions)
        #dstb_log_prob = th.vstack(dstb_log_prob)
        dstb_entropy = dstb_distribution[0].entropy()
        #dstb_entropy = th.hstack(dstb_entropy)
        #dstb_entropy = th.vstack(dstb_entropy)
        return dstb_log_prob, dstb_entropy

    def evaluate_states(self, obs, buf_num, env_indices=None, side_flag=None) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        if len(buf_num) != 1:
            assert self.num_adversaries > 1
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        features = self.vf_features_extractor(preprocessed_obs)
        if self.use_mirror:
            latent_vf = self.mlp_extractor.forward_critic(features, side_flag=side_flag)
        else:
            latent_vf = self.mlp_extractor.forward_critic(features)
        latents_per_adv = latent_vf.shape[0] // self.num_adversaries
        values = th.zeros((latent_vf.shape[0], 1), device=self.device)
        env_ids = env_indices // self.envs_per_matchup
        for i in range(len(buf_num)):
            indices = (env_ids == buf_num[i])
            if th.all(indices == False):
                # check to see if all the indices are False
                continue
            if len(indices.shape) > 1:
                indices = indices[:, 0]
            key = select_matchup_env(self.matchups, buf_num[i], self.envs_per_matchup)
            if len(buf_num) == 1:
                if isinstance(indices, np.ndarray):
                    indices = th.ones_like(th.from_numpy(indices))
                else:

                    indices = th.ones_like(indices)
            values[indices] = self.value_net[key](latent_vf[indices])
        #values = self.value_net(latent_vf)
        return values
    
    def evaluate_states_and_actions(self, obs, ego_actions, adv_actions, buf_num, env_indices=None):
        pass

    def predict(self, obs, deterministic=False, side_flag=None) -> Tuple[th.Tensor, th.Tensor]:
       if self.use_mirror:
           ego_actions, ego_log_prob = self.ego_forward(obs, deterministic, side_flag=side_flag)
           adv_side = (1 - side_flag) if side_flag is not None else None
           adv_actions, adv_log_prob = self.adv_forward(obs, deterministic, side_flag=adv_side)
       else:
           ego_actions, ego_log_prob = self.ego_forward(obs, deterministic)
           adv_actions, adv_log_prob = self.adv_forward(obs, deterministic)
       return (ego_actions, ego_log_prob), (adv_actions, adv_log_prob)
    
    def move_all_optimizers(self, device: torch.device) -> None:
        """This function moves all optimizers to the device."""
        for optimizer_name in ['value_optimizer', 'ctrl_optimizer', 'dstb_optimizer']:
            optimizer = getattr(self, optimizer_name, None)
            move_optimizer_to_device(optimizer, device)
