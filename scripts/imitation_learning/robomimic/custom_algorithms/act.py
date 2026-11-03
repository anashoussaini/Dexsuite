"""
Implementation of ACT (Action Chunking with Transformers).

Reference: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
           Zhao et al., RSS 2023

The L1 + KL loss and the temporal ensembling in get_action follow the original ACT
code (https://github.com/tonyzhaozh/act, policy.py and imitate_episodes.py),
Copyright (c) 2023 Tony Z. Zhao, MIT License (LICENSE-ACT in this folder). Modified
by the DexSuite authors, see NOTICE in this folder.

Lives here (not in a robomimic checkout) so pulling this branch is enough to run
ACT -- registry.py's register_custom_algorithms() auto-imports this file, which
triggers the @register_algo_factory_func/ALGO_NAME registration below.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import register_algo_factory_func, PolicyAlgo

from . import act_nets as ACTNets


def _robust_image_processor(obs):
    """
    Image processor that handles both [0, 255] inputs (from HDF5 datasets)
    and [0, 1] inputs (from DexSuite env which pre-normalizes).

    The default robomimic processor always divides by 255, which causes
    double-normalization when the env already returns [0, 1] images.
    """
    obs = TensorUtils.to_float(obs)
    # Check if images are already in [0, 1] range.  A max > 1 means
    # they are still in the uint8 / [0, 255] convention.
    if obs.max() > 1.0:
        obs = obs / 255.0
    obs = obs.clip(0.0, 1.0)
    obs = ObsUtils.batch_image_hwc_to_chw(obs)
    return obs


# Register once at import time so training and rollout use the same processor
# (only differs from the default by the conditional /255).
ObsUtils.ImageModality.set_obs_processor(processor=_robust_image_processor)


# hdf5_normalize_obs + cameras: SequenceDataset computes obs stats AFTER HWC->CHW,
# but RolloutPolicy normalizes BEFORE that conversion -- stats are CHW, rollout data
# is HWC, so normalize_dict's shape assertion fires the first time a rollout starts
# (training looks fine for however many epochs, since its own path is self-consistent).
# Fix is to never normalize image keys at all, not to reconcile the shapes -- images
# already have their own normalization (the /255 above, ImageNet stats in the
# pretrained backbone); scaling them again by per-pixel dataset stats would silently
# double-normalize. Also fixes bc/bc_transformer/diffusion_policy, which hit the same
# thing the moment they roll out with cameras and hdf5_normalize_obs on.
_orig_normalize_dict = ObsUtils.normalize_dict


def _is_image_like(key):
    # .get() (not key_is_obs_modality()) so action keys like "action/eef_pos" -- not in
    # OBS_KEYS_TO_MODALITIES -- pass through as False instead of getting silently
    # registered as low_dim with a warning printed per DataLoader worker per epoch.
    return ObsUtils.OBS_KEYS_TO_MODALITIES.get(key) in ("rgb", "depth")


def _normalize_skipping_images(dict, normalization_stats):
    """ObsUtils.normalize_dict, but rgb/depth keys pass through untouched."""
    to_norm = {k: v for k, v in dict.items() if not _is_image_like(k)}
    if len(to_norm) == len(dict):
        return _orig_normalize_dict(dict, normalization_stats)
    if not to_norm:
        return dict
    normed = _orig_normalize_dict(to_norm, normalization_stats)
    out = {k: v for k, v in dict.items()}
    out.update(normed)
    return out


ObsUtils.normalize_dict = _normalize_skipping_images


@register_algo_factory_func("act")
def algo_config_to_class(algo_config):
    return ACT, {}


class ACT(PolicyAlgo):
    """
    ACT training: a CVAE encoder compresses the action chunk into a latent z
    during training; the transformer decoder predicts the action chunk conditioned
    on observations and z. At test time z=0 and actions are returned one per call
    from a pre-computed chunk (action buffer).
    """

    def _create_networks(self):
        self.nets["policy"] = ACTNets.ActionChunkingTransformerActorNetwork(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            observation_horizon=self.algo_config.horizon.observation_horizon,
            action_horizon=self.algo_config.horizon.action_horizon,
            transformer_embed_dim=self.algo_config.transformer.embed_dim,
            transformer_num_layers=self.algo_config.transformer.num_layers,
            transformer_num_heads=self.algo_config.transformer.num_heads,
            transformer_feedforward_dim=self.algo_config.transformer.feedforward_dim,
            transformer_emb_dropout=self.algo_config.transformer.emb_dropout,
            transformer_attn_dropout=self.algo_config.transformer.attn_dropout,
            transformer_block_output_dropout=self.algo_config.transformer.block_output_dropout,
            transformer_sinusoidal_embedding=self.algo_config.transformer.sinusoidal_embedding,
            transformer_activation=self.algo_config.transformer.activation,
            transformer_nn_parameter_for_timesteps=self.algo_config.transformer.nn_parameter_for_timesteps,
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            latent_dim=self.algo_config.cvae.latent_dim,
            cvae_num_encoder_layers=self.algo_config.cvae.num_encoder_layers,
            detr_feedforward_dim=self.algo_config.transformer.feedforward_dim,
            backbone_pretrained=self.algo_config.backbone.pretrained,
            backbone_dilation=self.algo_config.backbone.dilation if "dilation" in self.algo_config.backbone else False,
            img_aug_crop_height=self.algo_config.augmentation.crop_height,
            img_aug_crop_width=self.algo_config.augmentation.crop_width,
            img_aug_color_jitter=self.algo_config.augmentation.color_jitter,
        )
        self._action_buffer = []
        self.nets = self.nets.float().to(self.device)

    def _create_optimizers(self):
        """
        Override base class to use separate learning rates for the ResNet
        backbone (lower) and everything else (higher), matching the original
        ACT paper (detr/main.py:80-88).
        """
        self.optimizers = dict()
        self.lr_schedulers = dict()
        self.step_lr_schedulers_every_batch = dict()

        for k in self.optim_params:
            if k not in self.nets:
                continue

            self.step_lr_schedulers_every_batch[k] = self.optim_params[k].learning_rate.get(
                "step_every_batch", False
            )

            if k == "policy":
                lr = self.optim_params[k]["learning_rate"]["initial"]
                wd = self.optim_params[k]["regularization"]["L2"]
                try:
                    lr_mult = self.algo_config.backbone.lr_multiplier
                except RuntimeError:
                    lr_mult = 1.0

                backbone_params = []
                non_backbone_params = []
                for name, param in self.nets[k].named_parameters():
                    if "backbone" in name:
                        backbone_params.append(param)
                    else:
                        non_backbone_params.append(param)

                param_groups = [
                    {"params": non_backbone_params},
                    {"params": backbone_params, "lr": lr * lr_mult},
                ]
                self.optimizers[k] = torch.optim.AdamW(
                    param_groups, lr=lr, weight_decay=wd
                )
            else:
                self.optimizers[k] = TorchUtils.optimizer_from_optim_params(
                    net_optim_params=self.optim_params[k], net=self.nets[k]
                )

            self.lr_schedulers[k] = TorchUtils.lr_scheduler_from_optim_params(
                net_optim_params=self.optim_params[k],
                net=self.nets[k],
                optimizer=self.optimizers[k],
            )

    def process_batch_for_training(self, batch):
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To] for k in batch["obs"]} # only use correct set of observations
        input_batch["goal_obs"] = batch.get("goal_obs", None)   #todo remove?
        input_batch["actions"] = batch["actions"][:, (To-1):(To-1+Ta)]  # future action chunk starting at current timestep
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def train_on_batch(self, batch, epoch, validate=False):
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super().train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch, validate=validate)
            losses = self._compute_losses(predictions, batch, epoch)
            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)
            if not validate:
                info.update(self._train_step(losses))
        return info

    def _cfg(self, path, default):
        """Read a nested algo_config value, falling back when an older config lacks the key.

        Config.__getitem__ raises when locked but returns an empty Config when unlocked, and an
        empty Config is falsy -- so a missing boolean key would otherwise read as False rather
        than as absent. Both misses are mapped to @default here.
        """
        node = self.algo_config
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return default if isinstance(node, dict) else node

    def _forward_training(self, batch, validate=False):
        try:
            cvae_enabled = self.algo_config.cvae.enabled
        except RuntimeError:
            cvae_enabled = True
        if validate or not cvae_enabled:
            # z=0 path: used for validation (to measure deployment-time L1) and when
            # CVAE is disabled entirely (z=0 at both train and test time, no KL loss).
            a_hat = self.nets["policy"](
                obs_dict=batch["obs"], goal_dict=batch["goal_obs"],
            )
            return OrderedDict(actions=a_hat)
        a_hat, mu, logvar = self.nets["policy"].forward_train(
            obs_dict=batch["obs"],
            actions=batch["actions"],
            goal_dict=batch["goal_obs"],
        )
        return OrderedDict(actions=a_hat, mu=mu, logvar=logvar)

    def _compute_losses(self, predictions, batch, epoch=0):
        l1_loss = F.l1_loss(predictions["actions"], batch["actions"])
        if "mu" not in predictions:
            # z=0 validation path — no posterior, no KL term.
            return OrderedDict(l1_loss=l1_loss, action_loss=l1_loss)
        mu, logvar = predictions["mu"], predictions["logvar"]
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
        warmup = self.algo_config.cvae.kl_warmup_epochs
        cycle = self.algo_config.cvae.kl_cycle_epochs
        if cycle > 0 and warmup > 0:
            epoch_in_cycle = epoch % cycle
            kl_weight = self.algo_config.cvae.kl_weight * min(1.0, epoch_in_cycle / warmup)
        elif warmup > 0:
            kl_weight = self.algo_config.cvae.kl_weight * min(1.0, epoch / warmup)
        else:
            kl_weight = self.algo_config.cvae.kl_weight
        action_loss = l1_loss + kl_weight * kl_loss
        return OrderedDict(l1_loss=l1_loss, kl_loss=kl_loss, action_loss=action_loss)

    def _train_step(self, losses):
        info = OrderedDict()
        info["policy_grad_norms"] = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
            max_grad_norm=self.global_config.train.max_grad_norm,
        )
        return info

    def get_action(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        Ta = self.algo_config.horizon.action_horizon

        # Query policy at every step and store the predicted chunk.
        action_chunk = self.nets["policy"](obs_dict, goal_dict=goal_dict)  # [1, Ta, ac_dim]
        if not self._cfg("temporal_ensemble.enabled", True):
            # No blending: execute the freshest prediction for the current step.
            return action_chunk[:, 0, :]
        self._action_buffer.append(action_chunk)
        if len(self._action_buffer) > Ta:
            self._action_buffer.pop(0)

        # Temporal ensembling: matches paper's imitate_episodes.py lines 251-259.
        # buffer[0]=oldest, buffer[n-1]=newest.
        # recency_bias=False (paper default): oldest chunk weighted highest.
        # recency_bias=True: newest chunk weighted highest.
        k = self.algo_config.temporal_ensemble.k
        recency_bias = self.algo_config.temporal_ensemble.recency_bias
        n = len(self._action_buffer)
        actions, weights = [], []
        for i in range(n):
            chunk = self._action_buffer[i]       # i=0 oldest, i=n-1 newest
            step_in_chunk = n - 1 - i            # oldest→last step, newest→step 0
            actions.append(chunk[:, step_in_chunk, :])
            decay_i = (n - 1 - i) if recency_bias else i
            weights.append(torch.exp(torch.tensor(-k * decay_i, dtype=torch.float32)))

        weights = torch.stack(weights).to(action_chunk.device)
        weights = weights / weights.sum()
        return sum(weights[i] * actions[i] for i in range(n))

    def reset(self):
        self._action_buffer = []

    def log_info(self, info):
        log = super().log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "kl_loss" in info["losses"]:
            log["KL_Loss"] = info["losses"]["kl_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
