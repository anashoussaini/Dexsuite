"""
Network architecture for ACT (Action Chunking with Transformers).

Reference: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
           Zhao et al., RSS 2023  --  https://github.com/tonyzhaozh/act

The policy predicts a chunk of future actions in one forward pass rather than a single
action. A CVAE encoder compresses the ground-truth action chunk into a latent z during
training; at test time z=0 and only the transformer decoder runs.

Adapted from the original ACT code (detr/models/detr_vae.py), Copyright (c) 2023
Tony Z. Zhao, MIT License (LICENSE-ACT in this folder), and from DETR
(https://github.com/facebookresearch/detr, models/transformer.py,
models/position_encoding.py and models/backbone.py), Copyright (c) Facebook, Inc. and
its affiliates, Apache License 2.0 (LICENSE-DETR in this folder). Modified by the
DexSuite authors, NOTICE in this folder lists the changes.

Kept in its own module rather than in policy_nets.py so that adding ACT to a robomimic
checkout touches no existing file beyond two __init__ registrations.
"""
import math
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.utils.tensor_utils as TensorUtils
from robomimic.models.base_nets import Module
from robomimic.models.obs_nets import ObservationGroupEncoder


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = torch.randn_like(std)
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    """
    Sinusoidal position encoding table.
    Ported from the original ACT codebase (detr_vae.py).
    Returns a fixed (non-trainable) table of shape [1, n_position, d_hid].
    """
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even dims: sin
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd dims: cos
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)  # [1, n_position, d_hid]


_DETR_ACTIVATIONS = {"relu": F.relu, "gelu": F.gelu, "glu": F.glu}


class _DetrEncoderLayer(nn.Module):
    """
    Transformer encoder layer that re-injects positional encodings into Q and K
    at every forward pass, matching the original DETR/ACT implementation
    (transformer.py TransformerEncoderLayer.forward_post lines 171-178).

    Positional info is added to Q and K but NOT to V. This keeps the spatial
    signal fresh at every attention layer rather than letting it dilute through
    residual connections from layer 0 only.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = _DETR_ACTIVATIONS[activation]

    def forward(self, src, pos=None):
        # pos added to Q and K, not V — matches transformer.py lines 171-172
        q = k = src if pos is None else src + pos
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class _DetrDecoderLayer(nn.Module):
    """
    Transformer decoder layer that re-injects positional encodings at every
    forward pass, matching the original DETR/ACT implementation
    (transformer.py TransformerDecoderLayer.forward_post lines 229-250).

    Self-attention:  query_pos added to Q and K, not V.
    Cross-attention: query_pos added to Q, pos (encoder spatial pos) added to K, not V.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.act = _DETR_ACTIVATIONS[activation]

    def forward(self, tgt, memory, pos=None, query_pos=None):
        # Self-attention: query_pos to Q and K, not V — transformer.py lines 236-237
        q = k = tgt if query_pos is None else tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # Cross-attention: query_pos to Q, encoder pos to K of memory, not V
        # — transformer.py lines 241-244
        tgt2 = self.cross_attn(
            query=tgt if query_pos is None else tgt + query_pos,
            key=memory if pos is None else memory + pos,
            value=memory,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # FFN
        tgt2 = self.linear2(self.dropout(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class PositionEmbeddingSine(nn.Module):
    """
    2D sinusoidal position encoding for image feature maps.
    Ported from the original DETR / ACT codebase (position_encoding.py).

    Each spatial position (row, col) in the feature map gets a unique encoding
    vector. Half the dimensions encode the y-position (row), the other half
    encode the x-position (column), using sin/cos at different frequencies —
    the same idea as 1D sinusoidal embeddings in "Attention Is All You Need",
    but extended to 2D so the transformer knows where each image patch came from.

    Args:
        num_pos_feats: half of the output dimension (output is num_pos_feats*2,
                       since half is used for x and half for y)
        temperature: controls the frequency range of the sinusoids
        normalize: if True, normalize positions to [0, 2*pi] range
        scale: normalization scale (defaults to 2*pi if normalize=True)
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is None and normalize:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        """
        Args:
            x: image feature tensor of shape [B, C, H, W]
        Returns:
            pos: position encoding of shape [1, num_pos_feats*2, H, W]
                 (batch dim is 1 because positions are the same for all images)
        """
        # Create cumulative position indices for rows (y) and columns (x)
        # not_mask is all ones — in original DETR this would mask padded regions,
        # but ACT doesn't use padding masks, so every position is valid
        not_mask = torch.ones(1, x.shape[2], x.shape[3], device=x.device, dtype=torch.float32)
        y_embed = not_mask.cumsum(1)  # [1, H, W] — row indices: 1, 2, ..., H
        x_embed = not_mask.cumsum(2)  # [1, H, W] — col indices: 1, 2, ..., W

        # Normalize to [0, 2*pi] range so positions are scale-invariant
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # Create frequency bands: temperature^(2i/D) for i = 0, 1, ..., D/2-1
        # Lower dimensions = lower frequency (coarse position), higher = finer
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Divide positions by frequency bands → [1, H, W, num_pos_feats]
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # Interleave sin and cos: even dims get sin, odd dims get cos
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # Concatenate y and x encodings → [1, H, W, num_pos_feats*2]
        # Then permute to channel-first: [1, num_pos_feats*2, H, W]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class ActionChunkingTransformerActorNetwork(Module):
    """
    DETR-style ACT policy: a bidirectional encoder processes observation tokens
    (images as spatial tokens + proprioception + latent z), and a decoder with
    cross-attention predicts action chunks from learned query embeddings.
    A CVAE encoder compresses the action sequence into a latent variable z during
    training; at test time z=0 is used.
    """

    def __init__(
        self,
        obs_shapes,
        ac_dim,
        observation_horizon,
        action_horizon,
        transformer_embed_dim,
        transformer_num_layers,
        transformer_num_heads,
        transformer_feedforward_dim=2048,
        transformer_emb_dropout=0.1,
        transformer_attn_dropout=0.1,
        transformer_block_output_dropout=0.1,
        transformer_sinusoidal_embedding=False,
        transformer_activation="gelu",
        transformer_nn_parameter_for_timesteps=False,
        goal_shapes=None,
        encoder_kwargs=None,
        latent_dim=32,
        cvae_num_encoder_layers=4,
        cvae_num_heads=8,
        cvae_feedforward_dim=2048,
        cvae_dropout=0.1,
        # DETR-specific parameters (with defaults so act.py needs no changes)
        detr_encoder_num_layers=4,
        detr_feedforward_dim=2048,
        backbone_pretrained=True,
        backbone_dilation=False,
        # Image augmentation applied directly in backbone path (bypasses obs encoder)
        img_aug_crop_height=0,   # 0 = no crop; front camera gets random crop, wrist gets center crop
        img_aug_crop_width=0,
        img_aug_color_jitter=False,
    ):
        super().__init__()

        assert isinstance(obs_shapes, OrderedDict)
        assert observation_horizon >= 1
        assert action_horizon >= 1

        self.obs_shapes = OrderedDict(obs_shapes)
        self.goal_shapes = OrderedDict(goal_shapes) if goal_shapes is not None else OrderedDict()
        self._is_goal_conditioned = len(self.goal_shapes) > 0

        self.ac_dim = ac_dim
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon
        self.latent_dim = latent_dim

        embed_dim = transformer_embed_dim
        self.embed_dim = embed_dim

        # Separate obs keys into image (3D: [C,H,W]) and low-dim (1D: [D,]).
        # Images bypass ObservationGroupEncoder and go through our own ResNet.
        # Low-dim obs go through ObservationGroupEncoder (which just concatenates them).
        self.image_keys = []
        self.low_dim_keys = []
        for k, shape in self.obs_shapes.items():
            if len(shape) == 3:  # [C, H, W] — image observation
                self.image_keys.append(k)
            else:                # [D,] — low-dim observation (joint pos, gripper, etc.)
                self.low_dim_keys.append(k)
        self.has_images = len(self.image_keys) > 0

        # Augmentation flags (set properly inside has_images block; defaults here for low-dim-only case)
        self._aug_crop_height = 0
        self._aug_crop_width = 0
        self._aug_color_jitter = False

        self.nets = nn.ModuleDict()
        self.params = nn.ParameterDict()

        # =====================================================================
        # Low-dim observation encoder
        # =====================================================================
        # ObservationGroupEncoder concatenates all low-dim obs into a single
        # vector (e.g. qpos[7] + grip_pos[3] + eef_quat[4] = 14-dim).
        # Used by both the CVAE encoder and the proprio projection below.
        # Images are excluded here — they get their own ResNet pipeline.
        low_dim_shapes = OrderedDict((k, self.obs_shapes[k]) for k in self.low_dim_keys)
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = low_dim_shapes
        if self._is_goal_conditioned:
            observation_group_shapes["goal"] = OrderedDict(self.goal_shapes)

        self.nets["encoder"] = ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
            feature_activation=None,
        )
        encoder_out_dim = self.nets["encoder"].output_shape()[0]

        # Project the concatenated proprio vector to transformer embedding dim.
        # This becomes the "proprio token" in the DETR encoder input.
        self.nets["proprio_proj"] = nn.Linear(encoder_out_dim, embed_dim)

        # =====================================================================
        # Image backbone (only created when RGB observations are present)
        # =====================================================================
        if self.has_images:
            import torchvision
            from torchvision.ops.misc import FrozenBatchNorm2d

            # ResNet18 conv layers (shared across all cameras, like original ACT).
            # children()[:-2] removes the final avgpool and fc layers, keeping
            # conv1 → bn1 → relu → maxpool → layer1 → layer2 → layer3 → layer4.
            # Output: [B, 512, h, w] where h,w depend on input image size.
            #
            # FrozenBatchNorm2d matches the original ACT backbone (backbone.py:94).
            # It preserves pretrained BN statistics instead of updating them during
            # fine-tuning, which is important when the backbone is pretrained.
            resnet = torchvision.models.resnet18(
                weights=(torchvision.models.ResNet18_Weights.DEFAULT if backbone_pretrained else None),
                replace_stride_with_dilation=[False, False, backbone_dilation],
                norm_layer=FrozenBatchNorm2d,
            )
            self.nets["backbone"] = nn.Sequential(*list(resnet.children())[:-2])

            # 1x1 conv projects ResNet's 512 channels down to embed_dim.
            # After this, each spatial position is an embed_dim-dimensional token.
            self.nets["input_proj"] = nn.Conv2d(512, embed_dim, kernel_size=1)

            # 2D sinusoidal position encoding so the transformer knows which
            # spatial position each image token came from (see PositionEmbeddingSine).
            self.position_encoding = PositionEmbeddingSine(
                num_pos_feats=embed_dim // 2, normalize=True
            )

            # ImageNet normalization constants (registered as buffers so they
            # move to the correct device automatically with .to(device)).
            self.register_buffer(
                'img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                'img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

            # ---- Image augmentation ----
            # Applied directly here since images bypass ObservationGroupEncoder
            # and its obs_randomizer pipeline entirely.
            # Front camera: random crop during training, center crop during eval.
            # Wrist camera: center crop always (no random translation).
            self._aug_crop_height = img_aug_crop_height
            self._aug_crop_width = img_aug_crop_width
            if img_aug_crop_height > 0:
                from robomimic.models.obs_core import CropRandomizer
                self.nets["img_aug_crop"] = CropRandomizer(
                    input_shape=(3, 224, 224),
                    crop_height=img_aug_crop_height,
                    crop_width=img_aug_crop_width,
                    num_crops=1,
                    pos_enc=False,
                )
            self._aug_color_jitter = img_aug_color_jitter
            if img_aug_color_jitter:
                from robomimic.models.obs_core import ColorRandomizer
                self.nets["img_aug_color"] = ColorRandomizer(
                    input_shape=(3, 1, 1),  # dummy — shape only used for output_shape(), not forward()
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1,
                    num_samples=1,
                )

        # =====================================================================
        # DETR encoder (processes observation context)
        # =====================================================================
        # Learned position embeddings for the z token (index 0) and the proprio
        # token (index 1). Image tokens use sinusoidal 2D position encoding
        # instead. Matches original ACT's self.additional_pos_embed.
        self.nets["additional_pos_embed"] = nn.Embedding(2, embed_dim)

        # Bidirectional self-attention encoder. Input is the sequence
        # [z_token, proprio_token, image_spatial_tokens...]. All tokens can
        # attend to all others — no causal mask. Uses _DetrEncoderLayer which
        # re-injects pos into Q and K at every layer, matching the original ACT
        # (transformer.py TransformerEncoderLayer.forward_post).
        self.nets["detr_encoder"] = nn.ModuleList([
            _DetrEncoderLayer(embed_dim, transformer_num_heads, transformer_feedforward_dim,
                              transformer_attn_dropout, transformer_activation)
            for _ in range(detr_encoder_num_layers)
        ])

        # =====================================================================
        # DETR decoder (predicts action chunk)
        # =====================================================================
        # The decoder uses cross-attention: learned action query tokens attend
        # to the encoder's output (memory) to extract the information they need.
        # Each query corresponds to one timestep in the action chunk.
        # Uses _DetrDecoderLayer which re-injects pos and query_pos into Q and K
        # at every layer, matching the original ACT
        # (transformer.py TransformerDecoderLayer.forward_post).
        self.nets["detr_decoder"] = nn.ModuleList([
            _DetrDecoderLayer(embed_dim, transformer_num_heads, transformer_feedforward_dim,
                              transformer_attn_dropout, transformer_activation)
            for _ in range(transformer_num_layers)
        ])
        self.nets["detr_decoder_norm"] = nn.LayerNorm(embed_dim)

        # =====================================================================
        # Action queries and output head
        # =====================================================================
        # Fixed sinusoidal position encoding for the action query tokens,
        # matching the original ACT paper (Fig. 11: "position embeddings (fixed)").
        # Registered as a buffer — moves to the right device with the model but
        # is not a trainable parameter.  Shape: [1, Ta, embed_dim].
        self.register_buffer(
            'query_pos_embed',
            get_sinusoid_encoding_table(action_horizon, embed_dim)
        )

        # Final linear layer: project each query's output to action dimensions
        self.nets["action_head"] = nn.Linear(embed_dim, ac_dim)

        # =====================================================================
        # CVAE encoder (training only — compresses action chunk into latent z)
        # =====================================================================
        # The CVAE encoder is a separate small transformer encoder that takes
        # [CLS_token, proprio_embedding, action_embeddings...] and outputs
        # (mu, logvar) from the CLS token. This is unchanged from before.
        # Note: the CVAE only sees proprioception, not images — matching the
        # original ACT paper which only feeds qpos to the CVAE encoder.
        self.params["cvae_cls"] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.nets["cvae_obs_proj"] = nn.Linear(encoder_out_dim, embed_dim)
        self.nets["cvae_action_proj"] = nn.Linear(ac_dim, embed_dim)
        # Uses _DetrEncoderLayer (same as DETR encoder) so that sinusoidal
        # positional encoding is re-injected into Q and K at every layer,
        # matching the original ACT paper's CVAE encoder (detr_vae.py:104).
        self.nets["cvae_encoder"] = nn.ModuleList([
            _DetrEncoderLayer(embed_dim, cvae_num_heads, cvae_feedforward_dim,
                              cvae_dropout, "relu")
            for _ in range(cvae_num_encoder_layers)
        ])
        # Sinusoidal positional encoding for CVAE encoder tokens:
        # [CLS, obs, action_0, ..., action_{Ta-1}] = 2 + Ta positions
        self.register_buffer(
            'cvae_pos_table',
            get_sinusoid_encoding_table(2 + action_horizon, embed_dim)  # [1, 2+Ta, D]
        )
        # Project CLS output to 2*latent_dim, then split into mu and logvar
        self.nets["cvae_latent_proj"] = nn.Linear(embed_dim, latent_dim * 2)
        # Project sampled latent z back to embed_dim for use as a token
        self.nets["cvae_latent_out_proj"] = nn.Linear(latent_dim, embed_dim)

        # =====================================================================
        # Parameter initialization
        # =====================================================================
        nn.init.normal_(self.params["cvae_cls"], mean=0.0, std=0.02)
        # Xavier uniform init for DETR encoder/decoder parameters, matching
        # the original ACT's Transformer._reset_parameters()
        for module in [self.nets["detr_encoder"], self.nets["detr_decoder"]]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def output_shape(self, input_shape=None):
        return [self.action_horizon, self.ac_dim]

    def _encode_cvae(self, obs_dict, actions):
        """
        Run the CVAE encoder on the current observation and the action chunk.
        Only uses low-dim (proprioception) observations, matching the original ACT.

        Args:
            obs_dict (dict): observation dict with leading dims [B, To, ...]
            actions (torch.Tensor): action chunk of shape [B, Ta, ac_dim]

        Returns:
            mu (torch.Tensor): [B, latent_dim]
            logvar (torch.Tensor): [B, latent_dim]
        """
        # Use current timestep (last, not first) and only low-dim obs
        obs_current = {k: obs_dict[k][:, -1:] for k in self.low_dim_keys}
        enc = TensorUtils.time_distributed(
            {"obs": obs_current, "goal": None}, self.nets["encoder"], inputs_as_kwargs=True
        )
        enc = enc[:, 0, :]  # [B, encoder_out_dim]

        B = enc.shape[0]
        obs_embed = self.nets["cvae_obs_proj"](enc).unsqueeze(1)   # [B, 1, D]
        action_embed = self.nets["cvae_action_proj"](actions)       # [B, Ta, D]
        cls = self.params["cvae_cls"].expand(B, -1, -1)             # [B, 1, D]

        encoder_input = torch.cat([cls, obs_embed, action_embed], dim=1)  # [B, 2+Ta, D]

        # Convert to sequence-first [2+Ta, B, D] for _DetrEncoderLayer,
        # which re-injects pos into Q and K at every layer (matching
        # the original ACT's CVAE encoder, detr_vae.py:100-104).
        encoder_input = encoder_input.permute(1, 0, 2)           # [2+Ta, B, D]
        cvae_pos = self.cvae_pos_table.permute(1, 0, 2)          # [2+Ta, 1, D]
        for layer in self.nets["cvae_encoder"]:
            encoder_input = layer(encoder_input, pos=cvae_pos)
        encoder_output = encoder_input.permute(1, 0, 2)          # [B, 2+Ta, D]
        cls_output = encoder_output[:, 0]                          # [B, D]

        latent_info = self.nets["cvae_latent_proj"](cls_output)  # [B, 2*latent_dim]
        mu = latent_info[:, :self.latent_dim]
        logvar = latent_info[:, self.latent_dim:]
        return mu, logvar

    def _decode(self, obs_dict, z_embed, goal_dict=None):
        """
        DETR-style decoding. The overall flow is:

        1. Encode proprioception (low-dim obs) into a single token
        2. Process each camera image through ResNet → 1x1 conv → spatial tokens
        3. Assemble all tokens: [z, proprio, cam1_spatial..., cam2_spatial...]
        4. Run bidirectional DETR encoder over all tokens (no causal mask)
        5. Run DETR decoder: action query tokens cross-attend to encoder output
        6. Project decoder outputs to action dimensions

        Only uses the current (most recent) timestep of observations, matching
        the original ACT paper. The observation_horizon of 10 from the config
        is not used here — only the current frame matters for this architecture.

        Args:
            obs_dict (dict): observation dict with leading dims [B, To, ...]
            z_embed (torch.Tensor): latent embedding [B, embed_dim]
                                    (sampled from CVAE during training, zeros at test time)
            goal_dict (dict): optional goal dict

        Returns:
            actions (torch.Tensor): predicted action chunk [B, Ta, ac_dim]
        """
        # Ensure a temporal dimension exists — during rollouts obs may arrive
        # without one if FrameStackWrapper isn't active yet
        # Pick a non-scalar key for the to_sequence check: scalars have
        # obs_shapes=() (len=0) and would cause the ndim comparison to fail.
        _cands = self.low_dim_keys or self.image_keys
        mod = next((k for k in _cands if len(self.obs_shapes.get(k, ())) > 0), _cands[0])
        if obs_dict[mod].ndim - 1 == len(self.obs_shapes[mod]):
            obs_dict = TensorUtils.to_sequence(obs_dict)

        # ---- Step 1: Proprioception ----
        # Take only the last timestep (current obs) and only low-dim keys.
        # ObservationGroupEncoder concatenates them into one flat vector
        # e.g. qpos[7] + grip_pos[3] + eef_quat[4] → [B, 14]
        current_obs = {k: obs_dict[k][:, -1:] for k in self.low_dim_keys}
        enc = TensorUtils.time_distributed(
            {"obs": current_obs, "goal": goal_dict}, self.nets["encoder"], inputs_as_kwargs=True
        )
        enc = enc[:, 0, :]  # [B, encoder_out_dim] — remove time dim
        proprio_token = self.nets["proprio_proj"](enc)  # [B, embed_dim]

        B = proprio_token.shape[0]

        # nn.Transformer uses sequence-first format: [seq_len, batch, embed_dim]
        z_token = z_embed.unsqueeze(0)       # [1, B, embed_dim]
        proprio_token = proprio_token.unsqueeze(0)  # [1, B, embed_dim]

        # Learned position embeddings for z (index 0) and proprio (index 1).
        # These let the encoder distinguish between token types, since z and
        # proprio don't have spatial positions the way image tokens do.
        additional_pos = self.nets["additional_pos_embed"].weight.unsqueeze(1).expand(-1, B, -1)  # [2, B, D]

        if self.has_images:
            # ---- Step 2: Process images through ResNet backbone ----
            all_cam_features = []
            all_cam_pos = []
            for cam_key in self.image_keys:
                # Current timestep image only: [B, C, H, W]
                img = obs_dict[cam_key][:, -1]

                # Crop augmentation: front camera gets random crop during training,
                # wrist camera always gets center crop (preserves close-up alignment).
                if self._aug_crop_height > 0:
                    is_wrist = "wrist" in cam_key
                    if self.training and not is_wrist:
                        img = self.nets["img_aug_crop"]._forward_in(img)
                    else:
                        img = self.nets["img_aug_crop"]._forward_in_eval(img)

                # Color jitter: both cameras, training only
                if self._aug_color_jitter and self.training:
                    img = self.nets["img_aug_color"]._forward_in(img)

                # Normalize to ImageNet statistics since ResNet expects this
                img = (img - self.img_mean) / self.img_std

                # ResNet18 conv layers: [B, C, H, W] → [B, 512, h, w]
                # h, w are the spatial dims after all downsampling
                # e.g. 128×128 input → roughly 4×4 feature map
                features = self.nets["backbone"](img)

                # 1x1 conv: [B, 512, h, w] → [B, embed_dim, h, w]
                # Each (i, j) position is now an embed_dim-dimensional token
                features = self.nets["input_proj"](features)

                # 2D sinusoidal pos encoding: [1, embed_dim, h, w]
                # Encodes (row, col) so the transformer knows where each token is
                pos = self.position_encoding(features)

                all_cam_features.append(features)
                all_cam_pos.append(pos)

            # Concatenate cameras along the width dimension, matching original
            # ACT (detr_vae.py line 129). Treats multiple cameras as one wide
            # feature map — heights must match across cameras.
            src = torch.cat(all_cam_features, dim=3)  # [B, embed_dim, h, w*num_cams]
            pos = torch.cat(all_cam_pos, dim=3)        # [1, embed_dim, h, w*num_cams]

            # Flatten spatial dims into a sequence, convert to sequence-first
            src = src.flatten(2).permute(2, 0, 1)                    # [H*W*cams, B, embed_dim]
            pos = pos.flatten(2).permute(2, 0, 1).expand(-1, B, -1)  # [H*W*cams, B, embed_dim]

            # Full encoder input: [z_token, proprio_token, img_token_0, ..., img_token_N]
            encoder_src = torch.cat([z_token, proprio_token, src], dim=0)
            encoder_pos = torch.cat([additional_pos, pos], dim=0)
        else:
            # Low-dim only (no images): encoder just sees z and proprio
            encoder_src = torch.cat([z_token, proprio_token], dim=0)  # [2, B, embed_dim]
            encoder_pos = additional_pos  # [2, B, embed_dim]

        # ---- Step 3: DETR Encoder ----
        # pos is re-injected into Q and K (not V) at every layer, matching the
        # original ACT transformer.py. The content stream (src) is passed as V
        # unchanged so positional info doesn't corrupt semantic content.
        memory = encoder_src
        for layer in self.nets["detr_encoder"]:
            memory = layer(memory, pos=encoder_pos)  # [S, B, embed_dim]

        # ---- Step 4: DETR Decoder ----
        # tgt starts as zeros. query_pos and encoder pos are re-injected into
        # Q and K at every layer — query_pos identifies which action timestep
        # each query is, encoder pos identifies where each memory token came from.
        # query_pos_embed: [1, Ta, D] → [Ta, B, D] (sequence-first, expanded over batch)
        query_pos = self.query_pos_embed.squeeze(0).unsqueeze(1).expand(-1, B, -1)
        tgt = torch.zeros(self.action_horizon, B, self.embed_dim, device=query_pos.device)

        for layer in self.nets["detr_decoder"]:
            tgt = layer(tgt, memory, pos=encoder_pos, query_pos=query_pos)
        hs = self.nets["detr_decoder_norm"](tgt)  # [Ta, B, embed_dim]

        # ---- Step 5: Predict actions ----
        hs = hs.permute(1, 0, 2)              # [B, Ta, embed_dim]
        actions = self.nets["action_head"](hs)  # [B, Ta, ac_dim]
        return actions

    def forward_train(self, obs_dict, actions, goal_dict=None):
        """
        Training forward pass. Uses the CVAE encoder to sample z from the action chunk,
        then decodes an action chunk conditioned on z.

        Args:
            obs_dict (dict): observation dict with leading dims [B, To, ...]
            actions (torch.Tensor): ground-truth action chunk [B, Ta, ac_dim]
            goal_dict (dict): optional goal dict

        Returns:
            a_hat (torch.Tensor): predicted action chunk [B, Ta, ac_dim]
            mu (torch.Tensor): CVAE encoder mean [B, latent_dim]
            logvar (torch.Tensor): CVAE encoder log-variance [B, latent_dim]
        """
        mu, logvar = self._encode_cvae(obs_dict, actions)
        z = reparametrize(mu, logvar)
        z_embed = self.nets["cvae_latent_out_proj"](z)
        a_hat = self._decode(obs_dict, z_embed, goal_dict=goal_dict)
        return a_hat, mu, logvar

    def forward(self, obs_dict, goal_dict=None):
        """
        Inference forward pass. Uses z=0 (no CVAE encoder).

        Args:
            obs_dict (dict): observation dict with leading dims [B, To, ...]
            goal_dict (dict): optional goal dict

        Returns:
            actions (torch.Tensor): predicted action chunk [B, Ta, ac_dim]
        """
        mod = self.low_dim_keys[0] if self.low_dim_keys else self.image_keys[0]
        B = obs_dict[mod].shape[0]
        z = torch.zeros(B, self.latent_dim, device=obs_dict[mod].device)
        z_embed = self.nets["cvae_latent_out_proj"](z)
        return self._decode(obs_dict, z_embed, goal_dict=goal_dict)

    def _to_string(self):
        return (
            f"action_dim={self.ac_dim}, obs_horizon={self.observation_horizon}, "
            f"action_horizon={self.action_horizon}"
        )
