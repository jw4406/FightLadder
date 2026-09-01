# Live results (auto-appended by run_results_watch.sh as each experiment finishes)

Experiments in flight:
- EXP-1: gamma=0.90 / lambda=0.80 / PopArt vs control, with a seed-1 noise floor
- EXP-2: unscaled reward (1.0 vs 0.001) -- does un-binding Adam eps help value EV
- EXP-3: aggresive_coeff=3 combat regime -- on-policy gamma paircorr

## EXP-1  gamma/lambda/PopArt vs seed-1 noise floor  (08-17 20:14)
```
REGIME SCREEN (value comparison valid only where regimes match):
  vctl               rew=-0.044 ego_ent=-0.963 adv_ent=-0.854
  s1_vctlB           rew=-0.044 ego_ent=-0.963 adv_ent=-0.854
  g0.90_vg090        rew=-0.0683 ego_ent=-0.902 adv_ent=-0.604
  lam0.80_vlam080    rew=-0.0231 ego_ent=-0.297 adv_ent=-0.376
  vpa(popart)        rew=-0.0654

VALUE HEAD_EV vs realised MC returns (100 eps, gamma 0.94, episode splits):
  vctl:    gamma  n_valid   HEAD EV  RIDGE EV      gap   std(V)   slope  |res|p50  |res|p95
  vctl:   HEAD  = the trained value head's own V(s), held-out episodes
  s1_vctlB:    gamma  n_valid   HEAD EV  RIDGE EV      gap   std(V)   slope  |res|p50  |res|p95
  s1_vctlB:   HEAD  = the trained value head's own V(s), held-out episodes
  g0.90_vg090:    gamma  n_valid   HEAD EV  RIDGE EV      gap   std(V)   slope  |res|p50  |res|p95
  g0.90_vg090:   HEAD  = the trained value head's own V(s), held-out episodes
  lam0.80_vlam080:    gamma  n_valid   HEAD EV  RIDGE EV      gap   std(V)   slope  |res|p50  |res|p95
  lam0.80_vlam080:   HEAD  = the trained value head's own V(s), held-out episodes
  READ: compare g090/lam080/vpa head_EV to vctl; a difference is real only
  if it exceeds |vctl - s1_vctlB| (the same-config seed noise floor).
```

# CLEAN RE-RUN  (08-17 23:17)  [--seed is inert; noise floor = eval variance]
```
## EXP-1  value head_EV, gamma 0.94 unless noted
-- NOISE FLOOR: same vctl policy, 3 eval seeds (spread = eval variance):
  floor_vctl_s0            head_EV=+0.1411  n=6479
  floor_vctl_s1            head_EV=+0.0285  n=6587
  floor_vctl_s2            head_EV=+0.0191  n=7157
-- TREATMENTS (eval seed 0, comparable to floor_vctl_s0):
  g090_at0.94              head_EV=+0.1892  n=4375
  g090_at0.90              head_EV=+0.1883  n=4375
  lam080                   head_EV=+0.3537  n=7027
  popart  FAILED
   NOTE: lam080 trained in a DIFFERENT regime (ego_ent -0.30 vs vctl -0.96);
   g090 at 0.90 has ceiling 0.69 not 0.59 -- not raw-comparable to vctl.

## EXP-2  unscaled reward: did un-binding Adam eps help value EV?
   mechanism already confirmed: value sqrt(v) 3.6e-9 -> 1.06e-5 (eps un-bound)
  rs001_scaled             head_EV=+0.1411  n=6479
  rs1_UNSCALED  FAILED
   EV is scale-free; rs1 above rs001 (beyond the exp1 noise band) => eps hurt V.

## EXP-3  aggresive_coeff=3 combat
-- WHO WINS (round outcomes; ep_rew is positive-sum at a=3, cannot say):
  a=1(vctl):   spar_Ry_Sa_11999808_steps.task  (260 rounds)
  a=1(vctl):     ego win   24.2%   ego lose  73.1%   draw   2.7%
  a=1(vctl):     end-of-round hp diff (agent - enemy) = -36.50 +/- 4.09
  a=1(vctl):     => ADVERSARY dominates by 48.8%.
  a=3(ac3):    spar_Ry_Sa_5999904_steps.task  (300 rounds)
  a=3(ac3):      ego win   12.0%   ego lose  87.3%   draw   0.7%
  a=3(ac3):      end-of-round hp diff (agent - enemy) = -92.75 +/- 3.93
  a=3(ac3):      => ADVERSARY dominates by 75.3%.
-- ON-POLICY gamma paircorr (combat regime learnable structure? baseline ~0.005):
  ON-POLICY, a=1 scoring: contact=5.2%  active=13  gamma_share=5.13%  |gamma|=8.9328  paircorr=-0.0197
  baseline (random play) paircorr ~ 0.005. engaged stalemate was 0.359.
  paircorr >> 0.005 => this policy's states have LEARNABLE joint structure.
  wrote headroom/clean_paircorr.json
```
DONE 23:25

# ===== ARM DIAGNOSTICS: ARM_A_unscaled_ac1  (reward_scale=1.0)  08-17 23:54 =====
```
-- REGIME (who wins; gates the value read):
    ego win   65.2%   ego lose  33.0%   draw   1.8%
    end-of-round hp diff (agent - enemy) = +18.62 +/- 4.09
    => EGO dominates by 32.1%.
-- INTRINSIC CEILING (EV_max for THIS policy; scale-invariant):
-- VALUE HEAD_EV (scale-matched) and the CONFOUND-ROBUST head_EV/EV_max:
-- ON-POLICY gamma paircorr (cross-state joint structure; baseline ~0.005):
  ON-POLICY, a=1 scoring: contact=0.8%  active=2  gamma_share=0.19%  |gamma|=0.7876  paircorr=+0.0592
-- FACTORED HEAD Q vs TRUE enumerated payoff (enumerating 300 states)...
  wrote /home/jw4406/codebase/FightLadder/main/headroom/diag_vclip_rs1.0_rs1_enum.json
       ckpt  states   ev_all     evW(M)     corrW(M)    evW(R)  corrW(R)            95% CI    CONST headroom
             CONST = a single fixed matrix for every state; HEADROOM = corrW(R) - CONST is the only part that is state-CONDITIONAL
-- Q ANOVA decomposition (mu/alpha/beta/gamma share of the head's Q):
  source EMULATOR PAYOFF r + gamma*V_scalar(s')   300 states   22x22 actions
  SS identity residual 0.00e+00   (must be ~0; orthogonality check)
  mu    (state)             90.0584%                 -
  alpha (ego main)           5.6566%          56.8980%
  beta  (adv main)           4.1410%          41.6535%
  gamma (INTERACTION)        0.1440%           1.4485%
  rms magnitudes   mu 5.711364   alpha 1.431380   beta 1.224707   gamma 0.228387
  gamma spectrum (mean normalised sv) 1.000 0.482 0.282 0.184
  gamma rank for 90% energy   median 3   p90 4
  gamma antisymmetric share 0.4637   (isotropic null 0.4762)
-- FACTORED-HEAD training metrics (final log dump):
  minimax_fx_gamma_share = 0.00355
  minimax_fx_w_norm = 6.64
  minimax_fx_anti_share = 0.462
  minimax_ev_ego = 0.938
  minimax_ev_adv = 0.967
  minimax_q_branch_std = 1.21
  minimax_target_corr = 0.983
  minimax_coverage = 1
```
ARM_DIAG_DONE ARM_A_unscaled_ac1 00:09

# ===== ARM DIAGNOSTICS: scaled_ac1_vctl  (reward_scale=0.001)  08-18 00:22 =====
```
-- REGIME (who wins; gates the value read):
    ego win   24.4%   ego lose  72.8%   draw   2.8%
    end-of-round hp diff (agent - enemy) = -35.94 +/- 4.19
    => ADVERSARY dominates by 48.4%.
-- INTRINSIC CEILING (EV_max for THIS policy; scale-invariant):
-- VALUE HEAD_EV (scale-matched) and the CONFOUND-ROBUST head_EV/EV_max:
  head_EV=+0.1356  EV_max=+nan  head_EV/EV_max=nan  (n=9030)
-- ON-POLICY gamma paircorr (cross-state joint structure; baseline ~0.005):
  ON-POLICY, a=1 scoring: contact=8.0%  active=20  gamma_share=0.97%  |gamma|=6.1873  paircorr=-0.0166
-- FACTORED HEAD Q vs TRUE enumerated payoff (enumerating 300 states)...
  wrote /home/jw4406/codebase/FightLadder/main/headroom/diag_vctl_enum.json
       ckpt  states   ev_all     evW(M)     corrW(M)    evW(R)  corrW(R)            95% CI    CONST headroom
             CONST = a single fixed matrix for every state; HEADROOM = corrW(R) - CONST is the only part that is state-CONDITIONAL
-- Q ANOVA decomposition (mu/alpha/beta/gamma share of the head's Q):
  source EMULATOR PAYOFF r + gamma*V_scalar(s')   300 states   22x22 actions
  SS identity residual 1.55e-16   (must be ~0; orthogonality check)
  mu    (state)             97.7438%                 -
  alpha (ego main)           0.9460%          41.9287%
  beta  (adv main)           1.1556%          51.2172%
  gamma (INTERACTION)        0.1546%           6.8541%
  rms magnitudes   mu 0.012423   alpha 0.001222   beta 0.001351   gamma 0.000494
  gamma spectrum (mean normalised sv) 1.000 0.475 0.257 0.159
  gamma rank for 90% energy   median 2   p90 4
  gamma antisymmetric share 0.3933   (isotropic null 0.4762)
-- FACTORED-HEAD training metrics (final log dump):
  minimax_fx_gamma_share = 0.0599
  minimax_fx_w_norm = 0.113
  minimax_fx_anti_share = 0.492
  minimax_ev_ego = 0.696
  minimax_ev_adv = 0.794
  minimax_q_branch_std = 0.00197
  minimax_target_corr = 0.892
  minimax_coverage = 1
```
ARM_DIAG_DONE scaled_ac1_vctl 00:41

# ===== ARM DIAGNOSTICS: ARM_B_unscaled_ac3  (reward_scale=1.0)  08-18 03:15 =====
```
-- REGIME (who wins; gates the value read):
    ego win   10.0%   ego lose  87.6%   draw   2.4%
    end-of-round hp diff (agent - enemy) = -88.09 +/- 3.91
    => ADVERSARY dominates by 77.6%.
-- INTRINSIC CEILING (EV_max for THIS policy; scale-invariant):
  [check] within-root var 4.196e-05  between-root var 2.026e-04
   gamma    Var(G|s)      Var(G)    EV_MAX  V head EV     K EV_max
  EV_MAX      ceiling for ANY value function against SINGLE-sample returns
              between EV_MAX and this is the prize for averaging targets
-- VALUE HEAD_EV (scale-matched) and the CONFOUND-ROBUST head_EV/EV_max:
  head_EV=-6.3978  EV_max=+0.8285  head_EV/EV_max=-7.72  (n=7189)
-- ON-POLICY gamma paircorr (cross-state joint structure; baseline ~0.005):
  ON-POLICY, a=1 scoring: contact=11.2%  active=28  gamma_share=1.15%  |gamma|=5.7619  paircorr=-0.0046
-- FACTORED HEAD Q vs TRUE enumerated payoff (enumerating 300 states)...
  wrote /home/jw4406/codebase/FightLadder/main/headroom/diag_vclip_rs1.0_ac3.0_unscAC3_11999808.json
       ckpt  states   ev_all     evW(M)     corrW(M)    evW(R)  corrW(R)            95% CI    CONST headroom
             CONST = a single fixed matrix for every state; HEADROOM = corrW(R) - CONST is the only part that is state-CONDITIONAL
 11,999,808     300   -0.196     -5.850        0.026   -74.285     0.002 [-0.004,+0.006]    0.003   -0.001
-- Q ANOVA decomposition (mu/alpha/beta/gamma share of the head's Q):
  source EMULATOR PAYOFF r + gamma*V_scalar(s')   300 states   22x22 actions
  SS identity residual 0.00e+00   (must be ~0; orthogonality check)
  mu    (state)             97.0759%                 -
  alpha (ego main)           2.2006%          75.2574%
  beta  (adv main)           0.6165%          21.0846%
  gamma (INTERACTION)        0.1070%           3.6581%
  rms magnitudes   mu 25.166637   alpha 3.789141   beta 2.005619   gamma 0.835395
  gamma spectrum (mean normalised sv) 1.000 0.503 0.290 0.182
  gamma rank for 90% energy   median 2   p90 4
  gamma antisymmetric share 0.4218   (isotropic null 0.4762)
-- FACTORED-HEAD training metrics (final log dump):
  minimax_fx_gamma_share = 0.027
  minimax_fx_w_norm = 22.7
  minimax_fx_anti_share = 0.472
  minimax_ev_ego = 0.609
  minimax_ev_adv = 0.712
  minimax_q_branch_std = 8.53
  minimax_target_corr = 0.844
  minimax_coverage = 1
```
ARM_DIAG_DONE ARM_B_unscaled_ac3 03:30

# ===== ARM DIAGNOSTICS: scaled_ac3_sclAC3  (reward_scale=0.001)  08-18 03:47 =====
```
-- REGIME (who wins; gates the value read):
    ego win   27.2%   ego lose  72.4%   draw   0.4%
    end-of-round hp diff (agent - enemy) = -60.71 +/- 4.60
    => ADVERSARY dominates by 45.2%.
-- INTRINSIC CEILING (EV_max for THIS policy; scale-invariant):
  [check] within-root var 6.873e-05  between-root var 2.178e-04
   gamma    Var(G|s)      Var(G)    EV_MAX  V head EV     K EV_max
  EV_MAX      ceiling for ANY value function against SINGLE-sample returns
              between EV_MAX and this is the prize for averaging targets
-- VALUE HEAD_EV (scale-matched) and the CONFOUND-ROBUST head_EV/EV_max:
  head_EV=-3.6506  EV_max=+0.7601  head_EV/EV_max=-4.80  (n=7238)
-- ON-POLICY gamma paircorr (cross-state joint structure; baseline ~0.005):
  ON-POLICY, a=1 scoring: contact=5.2%  active=13  gamma_share=8.11%  |gamma|=15.1512  paircorr=-0.0146
-- FACTORED HEAD Q vs TRUE enumerated payoff (enumerating 300 states)...
  wrote /home/jw4406/codebase/FightLadder/main/headroom/diag_ac3.0_sclAC3_11999808.json
       ckpt  states   ev_all     evW(M)     corrW(M)    evW(R)  corrW(R)            95% CI    CONST headroom
             CONST = a single fixed matrix for every state; HEADROOM = corrW(R) - CONST is the only part that is state-CONDITIONAL
 11,999,808     300    0.509     -1.694        0.066    -5.276     0.010 [-0.012,+0.032]    0.022   -0.012
-- Q ANOVA decomposition (mu/alpha/beta/gamma share of the head's Q):
  source EMULATOR PAYOFF r + gamma*V_scalar(s')   300 states   22x22 actions
  SS identity residual 0.00e+00   (must be ~0; orthogonality check)
  mu    (state)             97.2496%                 -
  alpha (ego main)           0.7483%          27.2069%
  beta  (adv main)           1.7987%          65.3981%
  gamma (INTERACTION)        0.2034%           7.3950%
  rms magnitudes   mu 0.014924   alpha 0.001309   beta 0.002030   gamma 0.000683
  gamma spectrum (mean normalised sv) 1.000 0.482 0.270 0.165
  gamma rank for 90% energy   median 2   p90 4
  gamma antisymmetric share 0.3345   (isotropic null 0.4762)
-- FACTORED-HEAD training metrics (final log dump):
  minimax_fx_gamma_share = 0.0078
  minimax_fx_w_norm = 0.0697
  minimax_fx_anti_share = 0.413
  minimax_ev_ego = 0.529
  minimax_ev_adv = 0.6
  minimax_q_branch_std = 0.00392
  minimax_target_corr = 0.778
  minimax_coverage = 1
```
ARM_DIAG_DONE scaled_ac3_sclAC3 03:58

# ===== LR-SWAP (adversary=slow leader) 08-18 13:57 =====
```
-- ROUND-END MODE (does the ADVERSARY now win by timeout, mirroring rs1?):

  spar_Ry_Sa_11999808_steps.task   (200 rounds)
   outcome |         KO    TIMEOUT |  row tot
       win |      26.0%      17.0% |    43.0%
      lose |      40.5%      16.0% |    56.5%
      draw |       0.0%       0.5% |     0.5%
     TOTAL |      66.5%      33.5% |

  on ego WINS (n=86): agent_hp   80.3+/-55.3   enemy_hp   27.2+/-47.5   margin  +53.1
    read: high both-hp + timeout => stall-to-timeout; low enemy_hp + KO => decisive win

  wrote headroom/rem_lrswap.json
```

# ===== ARM DIAGNOSTICS: LRSWAP_adv_leader  (reward_scale=1.0)  08-18 13:58 =====
```
-- REGIME (who wins; gates the value read):
    ego win   40.9%   ego lose  58.4%   draw   0.7%
    end-of-round hp diff (agent - enemy) = -13.62 +/- 5.50
    => ADVERSARY dominates by 17.5%.
-- INTRINSIC CEILING (EV_max for THIS policy; scale-invariant):
  [check] within-root var 3.238e-05  between-root var 7.982e-05
   gamma    Var(G|s)      Var(G)    EV_MAX  V head EV     K EV_max
  EV_MAX      ceiling for ANY value function against SINGLE-sample returns
              between EV_MAX and this is the prize for averaging targets
-- VALUE HEAD_EV (scale-matched) and the CONFOUND-ROBUST head_EV/EV_max:
  !! value_gap FAILED (see headroom/diag_vclip_rs1.0_lrswap_vg.err)
-- ON-POLICY gamma paircorr (cross-state joint structure; baseline ~0.005):
  ON-POLICY, a=1 scoring: contact=14.4%  active=36  gamma_share=2.25%  |gamma|=14.1786  paircorr=+0.0108
-- FACTORED HEAD Q vs TRUE enumerated payoff (enumerating 300 states)...
  wrote /home/jw4406/codebase/FightLadder/main/headroom/diag_vclip_rs1.0_lrswap_11999808.json
       ckpt  states   ev_all     evW(M)     corrW(M)    evW(R)  corrW(R)            95% CI    CONST headroom
             CONST = a single fixed matrix for every state; HEADROOM = corrW(R) - CONST is the only part that is state-CONDITIONAL
 11,999,808     300    0.381     -0.178        0.023    -0.182    -0.015 [-0.030,-0.000]    0.009   -0.024
-- Q ANOVA decomposition (mu/alpha/beta/gamma share of the head's Q):
  source EMULATOR PAYOFF r + gamma*V_scalar(s')   300 states   22x22 actions
  SS identity residual 0.00e+00   (must be ~0; orthogonality check)
  mu    (state)             84.5179%                 -
  alpha (ego main)           2.4848%          16.0492%
  beta  (adv main)          12.5547%          81.0917%
  gamma (INTERACTION)        0.4427%           2.8591%
  rms magnitudes   mu 7.540562   alpha 1.292918   beta 2.906250   gamma 0.545708
  gamma spectrum (mean normalised sv) 1.000 0.508 0.306 0.192
  gamma rank for 90% energy   median 3   p90 4
  gamma antisymmetric share 0.4500   (isotropic null 0.4762)
-- FACTORED-HEAD training metrics (final log dump):
  minimax_fx_gamma_share = 0.00735
  minimax_fx_w_norm = 9.42
  minimax_fx_anti_share = 0.482
  minimax_ev_ego = 0.813
  minimax_ev_adv = 0.871
  minimax_q_branch_std = 1.4
  minimax_target_corr = 0.934
  minimax_coverage = 1
```
ARM_DIAG_DONE LRSWAP_adv_leader 14:08

# ===== LBR EXPLOITABILITY over long50 milestones (stride 2, 20 eps) =====

# ===== LO-LR (ego 1e-5 slow leader; tests relative-vs-absolute) 08-18 21:47 =====
```
-- ROUND-END MODE (does the ego STILL win as slow leader at 1e-5?):

  spar_Ry_Sa_11999808_steps.task   (200 rounds)
   outcome |         KO    TIMEOUT |  row tot
       win |      16.5%       0.0% |    16.5%
      lose |      83.0%       0.0% |    83.0%
      draw |       0.5%       0.0% |     0.5%
     TOTAL |     100.0%       0.0% |

  on ego WINS (n=33): agent_hp   63.7+/-49.8   enemy_hp   -1.0+/-0.0   margin  +64.7
    read: high both-hp + timeout => stall-to-timeout; low enemy_hp + KO => decisive win

  wrote headroom/rem_lolr.json
-- final ego/adv entropy (collapsed?):
  ego_entropy_loss = -0.664
  adv_entropy_loss = -0.668
  ep_len_mean = 261
```

# ===== ARM DIAGNOSTICS: LOLR_ego_slowleader_1e5  (reward_scale=1.0)  08-18 21:48 =====
```
-- REGIME (who wins; gates the value read):
    ego win   18.2%   ego lose  81.4%   draw   0.4%
    end-of-round hp diff (agent - enemy) = -73.50 +/- 5.38
    => ADVERSARY dominates by 63.2%.
-- INTRINSIC CEILING (EV_max for THIS policy; scale-invariant):
  [check] within-root var 1.066e-04  between-root var 1.587e-04
   gamma    Var(G|s)      Var(G)    EV_MAX  V head EV     K EV_max
  EV_MAX      ceiling for ANY value function against SINGLE-sample returns
              between EV_MAX and this is the prize for averaging targets
-- VALUE HEAD_EV (scale-matched) and the CONFOUND-ROBUST head_EV/EV_max:
  head_EV=-0.2165  EV_max=+0.5981  head_EV/EV_max=-0.36  (n=9718)
-- ON-POLICY gamma paircorr (cross-state joint structure; baseline ~0.005):
  ON-POLICY, a=1 scoring: contact=6.0%  active=15  gamma_share=2.79%  |gamma|=6.7449  paircorr=-0.0089
-- FACTORED HEAD Q vs TRUE enumerated payoff (enumerating 300 states)...
  wrote /home/jw4406/codebase/FightLadder/main/headroom/diag_vclip_rs1.0_lolr_11999808.json
       ckpt  states   ev_all     evW(M)     corrW(M)    evW(R)  corrW(R)            95% CI    CONST headroom
             CONST = a single fixed matrix for every state; HEADROOM = corrW(R) - CONST is the only part that is state-CONDITIONAL
 11,999,808     300    0.659     -1.260        0.036    -1.831    -0.002 [-0.013,+0.010]    0.013   -0.014
-- Q ANOVA decomposition (mu/alpha/beta/gamma share of the head's Q):
  source EMULATOR PAYOFF r + gamma*V_scalar(s')   300 states   22x22 actions
  SS identity residual 1.92e-16   (must be ~0; orthogonality check)
  mu    (state)             97.1105%                 -
  alpha (ego main)           0.7580%          26.2323%
  beta  (adv main)           1.9695%          68.1598%
  gamma (INTERACTION)        0.1620%           5.6079%
  rms magnitudes   mu 11.391157   alpha 1.006391   beta 1.622231   gamma 0.465316
  gamma spectrum (mean normalised sv) 1.000 0.492 0.288 0.179
  gamma rank for 90% energy   median 2   p90 4
  gamma antisymmetric share 0.3900   (isotropic null 0.4762)
-- FACTORED-HEAD training metrics (final log dump):
  minimax_fx_gamma_share = 0.0109
  minimax_fx_w_norm = 15.3
  minimax_fx_anti_share = 0.497
  minimax_ev_ego = 0.842
  minimax_ev_adv = 0.875
  minimax_q_branch_std = 2.06
  minimax_target_corr = 0.935
  minimax_coverage = 1
```
ARM_DIAG_DONE LOLR_ego_slowleader_1e5 22:04

## long50 @ ~9M  (08-18 22:35)
```
[LBR] 1 matchup(s), 2 direction(s)
[LBR] ===== matchup 0: RyuVsSagat =====
[LBR] -- direction: LBR plays the ADV seat (ego_side=left, sgn=-1, gamma=0.94)
[LBR]    lbr      return=+0.01853 (20 eps, 93.53s)
[LBR]    selfplay return=-0.00661
[LBR] -- direction: LBR plays the EGO seat (ego_side=left, sgn=+1, gamma=0.94)
[LBR]    lbr      return=+0.01081 (20 eps, 97.98s)
[LBR]    selfplay return=+0.00661
DUALITY GAP (NashConv) per matchup -- LOWER BOUNDS
    lbr      eps_ego=+0.00420  eps_adv=+0.02515   ->  NashConv=+0.02935
  Note: NashConv = eps_ego + eps_adv (a SUM of both players' deviation
  on the true duality gap. eps < 0 means that seat's bound is vacuous.
```
LBR_MILESTONE_DONE 9M 22:39

# ===== ARM DIAGNOSTICS: lr155_49M  (reward_scale=1.0)  08-20 00:13 =====
```
-- REGIME (who wins; gates the value read):
    ego win   16.0%   ego lose  84.0%   draw   0.0%
    end-of-round hp diff (agent - enemy) = -45.34 +/- 3.26
    => ADVERSARY dominates by 68.0%.
-- INTRINSIC CEILING (EV_max for THIS policy; scale-invariant):
  [check] within-root var 5.809e-05  between-root var 2.076e-04
   gamma    Var(G|s)      Var(G)    EV_MAX  V head EV     K EV_max
  EV_MAX      ceiling for ANY value function against SINGLE-sample returns
              between EV_MAX and this is the prize for averaging targets
-- VALUE HEAD_EV (scale-matched) and the CONFOUND-ROBUST head_EV/EV_max:
  head_EV=+0.4536  EV_max=+0.7813  head_EV/EV_max=0.58  (n=8350)
-- ON-POLICY gamma paircorr (cross-state joint structure; baseline ~0.005):
  ON-POLICY, a=1 scoring: contact=9.6%  active=24  gamma_share=5.73%  |gamma|=14.0061  paircorr=+0.0207
-- FACTORED HEAD Q vs TRUE enumerated payoff (enumerating 300 states)...
  wrote /home/jw4406/codebase/FightLadder/main/headroom/diag_vclip_rs1.0_lr155_48999216.json
       ckpt  states   ev_all     evW(M)     corrW(M)    evW(R)  corrW(R)            95% CI    CONST headroom
             CONST = a single fixed matrix for every state; HEADROOM = corrW(R) - CONST is the only part that is state-CONDITIONAL
 48,999,216     300    0.534     -2.392        0.029    -3.832     0.008 [-0.011,+0.024]    0.010   -0.002
-- Q ANOVA decomposition (mu/alpha/beta/gamma share of the head's Q):
  source EMULATOR PAYOFF r + gamma*V_scalar(s')   300 states   22x22 actions
  SS identity residual 1.61e-16   (must be ~0; orthogonality check)
  mu    (state)             94.9255%                 -
  alpha (ego main)           0.7074%          13.9396%
  beta  (adv main)           4.1802%          82.3777%
  gamma (INTERACTION)        0.1869%           3.6827%
  rms magnitudes   mu 12.295506   alpha 1.061391   beta 2.580207   gamma 0.545549
  gamma spectrum (mean normalised sv) 1.000 0.528 0.320 0.209
  gamma rank for 90% energy   median 3   p90 4
  gamma antisymmetric share 0.3858   (isotropic null 0.4762)
-- FACTORED-HEAD training metrics (final log dump):
  minimax_fx_gamma_share = 0.00646
  minimax_fx_w_norm = 36.8
  minimax_fx_anti_share = 0.48
  minimax_ev_ego = 0.916
  minimax_ev_adv = 0.956
  minimax_q_branch_std = 3.9
  minimax_target_corr = 0.978
  minimax_coverage = 1
```
ARM_DIAG_DONE lr155_49M 00:31

# ===== ARM DIAGNOSTICS: lr155_38M  (reward_scale=1.0)  08-20 09:17 =====
```
-- REGIME (who wins; gates the value read):
    ego win   23.2%   ego lose  72.8%   draw   4.0%
    end-of-round hp diff (agent - enemy) = -15.38 +/- 2.62
    => ADVERSARY dominates by 49.6%.
-- INTRINSIC CEILING (EV_max for THIS policy; scale-invariant):
  [check] within-root var 3.397e-05  between-root var 3.126e-04
   gamma    Var(G|s)      Var(G)    EV_MAX  V head EV     K EV_max
  EV_MAX      ceiling for ANY value function against SINGLE-sample returns
              between EV_MAX and this is the prize for averaging targets
-- VALUE HEAD_EV (scale-matched) and the CONFOUND-ROBUST head_EV/EV_max:
  head_EV=+0.6575  EV_max=+0.9020  head_EV/EV_max=0.73  (n=7562)
-- ON-POLICY gamma paircorr (cross-state joint structure; baseline ~0.005):
  ON-POLICY, a=1 scoring: contact=1.6%  active=4  gamma_share=32.95%  |gamma|=13.1500  paircorr=+0.1001
-- FACTORED HEAD Q vs TRUE enumerated payoff (enumerating 300 states)...
  wrote /home/jw4406/codebase/FightLadder/main/headroom/diag_vclip_rs1.0_lr155_37999392.json
       ckpt  states   ev_all     evW(M)     corrW(M)    evW(R)  corrW(R)            95% CI    CONST headroom
             CONST = a single fixed matrix for every state; HEADROOM = corrW(R) - CONST is the only part that is state-CONDITIONAL
 37,999,392     300    0.526     -1.613        0.009    -2.393     0.008 [-0.003,+0.015]    0.006   +0.002
-- Q ANOVA decomposition (mu/alpha/beta/gamma share of the head's Q):
  source EMULATOR PAYOFF r + gamma*V_scalar(s')   300 states   22x22 actions
  SS identity residual 2.15e-16   (must be ~0; orthogonality check)
  mu    (state)             94.8712%                 -
  alpha (ego main)           0.6822%          13.3004%
  beta  (adv main)           4.2276%          82.4275%
  gamma (INTERACTION)        0.2191%           4.2722%
  rms magnitudes   mu 15.036398   alpha 1.275022   beta 3.174110   gamma 0.722621
  gamma spectrum (mean normalised sv) 1.000 0.495 0.258 0.160
  gamma rank for 90% energy   median 2   p90 3
  gamma antisymmetric share 0.4384   (isotropic null 0.4762)
-- FACTORED-HEAD training metrics (final log dump):
  minimax_fx_gamma_share = 0.00429
  minimax_fx_w_norm = 25.9
  minimax_fx_anti_share = 0.447
  minimax_ev_ego = 0.84
  minimax_ev_adv = 0.906
  minimax_q_branch_std = 2.63
  minimax_target_corr = 0.952
  minimax_coverage = 1
```
ARM_DIAG_DONE lr155_38M 09:27
