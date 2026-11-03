"""
Config for ACT (Action Chunking with Transformers) algorithm.
"""

from robomimic.config.base_config import BaseConfig


class ACTConfig(BaseConfig):
    ALGO_NAME = "act"

    def train_config(self):
        super().train_config()
        self.train.hdf5_load_next_obs = False
        self.train.seq_length = 8    # must match algo.horizon.action_horizon
        self.train.frame_stack = 1   # DETR arch uses only current frame; must match algo.horizon.observation_horizon

    def algo_config(self):
        # optimizer
        self.algo.optim_params.policy.optimizer_type = "adamw"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1
        self.algo.optim_params.policy.learning_rate.epoch_schedule = []
        self.algo.optim_params.policy.learning_rate.scheduler_type = "multistep"
        self.algo.optim_params.policy.learning_rate.do_not_lock_keys()
        self.algo.optim_params.policy.do_not_lock_keys()
        self.algo.optim_params.policy.regularization.L2 = 0.0

        # horizons
        self.algo.horizon.observation_horizon = 1
        self.algo.horizon.action_horizon = 8

        # CVAE encoder
        self.algo.cvae.enabled = True       # set False to use z=0 at both train and test time (no KL loss)
        self.algo.cvae.latent_dim = 32
        self.algo.cvae.kl_weight = 10.0   # from ACT paper
        self.algo.cvae.num_encoder_layers = 4
        self.algo.cvae.kl_warmup_epochs = 0  # ramp kl_weight linearly from 0 → kl_weight over this many epochs; 0 = no warmup
        self.algo.cvae.kl_cycle_epochs = 0   # repeat the warmup ramp every N epochs (cyclical annealing); 0 = no cycling

        # image backbone
        self.algo.backbone.pretrained = True   # must match original ACT (backbone.py:94)
        self.algo.backbone.lr_multiplier = 0.1  # backbone LR = main LR * this (original ACT: 1e-5 / 1e-4)
        self.algo.backbone.dilation = False    # replace layer4 stride with dilation → 14×14 tokens (vs 7×7)

        # image augmentation (applied directly in backbone path, bypasses obs encoder)
        self.algo.augmentation.crop_height = 0      # crop height in pixels; 0 = no crop (use full image)
        self.algo.augmentation.crop_width = 0       # crop width in pixels; 0 = no crop
        self.algo.augmentation.color_jitter = False  # random color jitter during training

        # temporal ensembling
        self.algo.temporal_ensemble.enabled = True        # False = execute the newest chunk's first action, no blending
        self.algo.temporal_ensemble.k = 0.01              # exponential decay rate
        self.algo.temporal_ensemble.recency_bias = False  # False = paper default (oldest chunk weighted highest); True = newest chunk weighted highest

        # transformer decoder
        self.algo.transformer.embed_dim = 512
        self.algo.transformer.num_layers = 6
        self.algo.transformer.num_heads = 8
        self.algo.transformer.feedforward_dim = 2048
        self.algo.transformer.emb_dropout = 0.1
        self.algo.transformer.attn_dropout = 0.1
        self.algo.transformer.block_output_dropout = 0.1
        self.algo.transformer.sinusoidal_embedding = False
        self.algo.transformer.activation = "gelu"
        self.algo.transformer.nn_parameter_for_timesteps = True
