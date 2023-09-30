from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import zero_module, convert_module_to_f16, convert_module_to_f32


class PosEncoding(nn.Module):
    """
    A positional encoding layer that adds a sinusoidal encoding to the input.

    Args:
        dim: the dimension of the input.
        max_freq: controls the minimum frequency of the embeddings.

    Returns:
        an [N x dim] Tensor of positional embeddings.
    """
    def __init__(self, dim, max_freq=10000):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        self.max_freq = max_freq
        half_dim = dim // 2
        self.register_buffer("freqs", torch.exp(-np.log(max_freq) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim))

    def forward(self, timesteps):
        args = timesteps[:, None] * self.freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding


class GroupNorm32(nn.GroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
   

class ModModule(nn.Module):
    """
    Any module where forward() takes modulation embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` modulation embeddings.
        """


class ModSequential(nn.Sequential, ModModule):
    """
    A sequential module that passes modulation embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, ModModule):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample2d(nn.Module):
    """
    An upsampling layer with an optional convolution.

    Args:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
    """
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)
        else:
            assert self.channels == self.out_channels, "channels must match if no conv"

    def forward(self, x):
        assert x.shape[1] == self.channels, f"channels must match, got input shape {x.shape} and channels {self.channels}"
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample2d(nn.Module):
    """
    A downsampling layer with an optional convolution.

    Args:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
    """
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels, "channels must match if no conv"
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels, f"channels must match, got input shape {x.shape} and channels {self.channels}"
        return self.op(x)


class ResBlock2d(ModModule):
    """
    A residual block that can optionally change the number of channels.

    Args:
        channels: channels in the inputs and outputs.
        emb_channels: channels in the modulation embeddings.
        dropout: dropout rate.
        out_channels: if specified, the number of output channels.
        num_groups: the number of groups for the group norm.
        use_conv: if True and out_channels is specified, use a spatial
            convolution instead of a smaller 1x1 convolution to change the
            channels in the skip connection.
        use_scale_shift_norm: use a FiLM-like conditioning mechanism.
        up: if True, use this block for upsampling.
        down: if True, use this block for downsampling.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        num_groups=32,
        use_conv=False,
        use_scale_shift_norm=True,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            GroupNorm32(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample2d(channels, False)
            self.x_upd = Upsample2d(channels, False)
        elif down:
            self.h_upd = Downsample2d(channels, False)
            self.x_upd = Downsample2d(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        self.out_layers = nn.Sequential(
            GroupNorm32(num_groups, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: an [N x C x ...] Tensor of features.
            emb: an [N x emb_channels] Tensor of timestep embeddings.

        Returns:
            an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        Args:
            qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.

        Returns:
            an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0, "QKV width must be divisible by 3 * n_heads"
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / np.sqrt(np.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    """
    def __init__(
        self,
        channels,
        num_groups=32,
        num_heads=1,
        num_head_channels=-1,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (channels % num_head_channels == 0), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = GroupNorm32(num_groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class AdmUnet2d(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    Args:
        in_channels: channels in the input Tensor.
        model_channels: base channel count for the model.
        out_channels: channels in the output Tensor.
        num_res_blocks: number of residual blocks per downsample.
        attention_resolutions: a collection of downsample rates at which
            attention will take place. May be a set, list, or tuple.
            For example, if this contains 4, then at 4x downsampling, attention
            will be used.
        dropout: the dropout probability.
        channel_mult: channel multiplier for each level of the UNet.
        conv_resample: if True, use learned convolutions for upsampling and
            downsampling.
        num_classes: if specified (as an int), then this model will be
            class-conditional with `num_classes` classes.
        has_null_class: if True, then label -1 will be treated as a
            addtionally null class.
        use_fp16: if True, use FP16 for the model.
        num_groups: the number of groups in each group norm layer.
        num_heads: the number of attention heads in each attention layer.
        num_heads_channels: if specified, ignore num_heads and instead use
            a fixed channel width per attention head.
        use_scale_shift_norm: use a FiLM-like conditioning mechanism.
        resblock_updown: use residual blocks for up/downsampling.
    """
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        num_classes = None,
        has_null_class = False,
        use_fp16=False,
        num_groups=32,
        num_heads=1,
        num_head_channels=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.has_null_class = has_null_class if num_classes is not None else False
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_groups = num_groups
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            PosEncoding(model_channels),
            nn.Linear(model_channels, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [ModSequential(nn.Conv2d(in_channels, ch, 3, padding=1))]
        )
        ds = image_size
        input_block_chs = [ch]
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock2d(
                        ch,
                        embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        num_groups=num_groups,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_groups=num_groups,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
                self.input_blocks.append(ModSequential(*layers))
                input_block_chs.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    ModSequential(
                        ResBlock2d(
                            ch,
                            embed_dim,
                            dropout,
                            out_channels=ch,
                            num_groups=num_groups,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample2d(
                            ch, conv_resample, out_channels=ch
                        )
                    )
                )
                input_block_chs.append(ch)
                ds //= 2

        self.middle_block = ModSequential(
            ResBlock2d(
                ch,
                embed_dim,
                dropout,
                num_groups=num_groups,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_groups=num_groups,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
            ),
            ResBlock2d(
                ch,
                embed_dim,
                dropout,
                num_groups=num_groups,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock2d(
                        ch + input_block_chs.pop(),
                        embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        num_groups=num_groups,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_groups=num_groups,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(
                        ResBlock2d(
                            ch,
                            embed_dim,
                            dropout,
                            out_channels=ch,
                            num_groups=num_groups,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample2d(
                            ch, conv_resample, out_channels=ch
                        )
                    )
                    ds *= 2
                self.output_blocks.append(ModSequential(*layers))

        self.out = nn.Sequential(
            GroupNorm32(num_groups, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channels, 3, padding=1)),
        )

        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self):
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    @property
    def example_inputs(self):
        """
        Return example inputs for model summary.
        """
        return {
            "x": torch.randn(1, self.in_channels, self.image_size, self.image_size).to(self.device),
            "times": torch.zeros((1,), dtype=torch.long).to(self.device),
            "classes": torch.randint(0, self.num_classes, (1,)).to(self.device) if self.num_classes is not None else None,
        } 

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, times, classes=None):
        """
        Apply the model to an input batch.

        Args:
            x: an [N x C x H x W] Tensor of inputs.
            times: an [N] tensor of times.
            classes: an [N] Tensor of labels, if class-conditional.
                Set indices to -1 for null class if supported. 
                None if not class-conditioned or if all samples are null class.

        Returns:
            an [N x C x H * W] Tensor of outputs.
        """
        assert classes is None or self.num_classes is not None, "this model is not class-conditioned"
        if classes is not None:
            assert torch.all(classes >= 0) or self.has_null_class, "this model does not have a null class"
        
        hs = []
        emb = self.time_embed(times)

        if self.num_classes is not None:
            if classes is not None:
                assert classes.shape == (x.shape[0],), "classes must be a 1-D batch of labels"
                class_emb = self.label_emb(classes * (classes >= 0).long())
                if self.has_null_class:
                    class_emb = class_emb * (classes >= 0).unsqueeze(1)
            else:
                class_emb = torch.zeros(x.shape[0], self.label_emb.weight.shape[1], device=x.device)
            emb = emb + class_emb

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)

