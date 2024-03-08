import streamlit as st
import os
import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#pip install torchaudio
import torchaudio



def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    groups = 32
    if channels <= 16:
        groups = 8
    elif channels <= 64:
        groups = 16
    while channels % groups != 0:
        groups = int(groups / 2)
    assert groups > 2
    return GroupNorm32(groups, channels)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/output heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None, rel_pos=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        if rel_pos is not None:
            weight = rel_pos(weight.reshape(bs, self.n_heads, weight.shape[-2], weight.shape[-1])).reshape(bs * self.n_heads, weight.shape[-2], weight.shape[-1])
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        if mask is not None:
            # The proper way to do this is to mask before the softmax using -inf, but that doesn't work properly on CPUs.
            mask = mask.repeat(self.n_heads, 1).unsqueeze(1)
            weight = weight * mask
        a = torch.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        do_checkpoint=True,
        relative_pos_embeddings=False,
    ):
        super().__init__()
        self.channels = channels
        self.do_checkpoint = do_checkpoint
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        # split heads before split qkv
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))
        if relative_pos_embeddings:
            self.relative_pos_embeddings = RelativePositionBias(scale=(channels // self.num_heads) ** .5, causal=False, heads=num_heads, num_buckets=32, max_distance=64)
        else:
            self.relative_pos_embeddings = None

    def forward(self, x, mask=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv, mask, self.relative_pos_embeddings)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channels=None, factor=4):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.factor = factor
        if use_conv:
            ksize = 5
            pad = 2
            self.conv = nn.Conv1d(self.channels, self.out_channels, ksize, padding=pad)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=self.factor, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channels=None, factor=4, ksize=5, pad=2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        stride = factor
        if use_conv:
            self.op = nn.Conv1d(
                self.channels, self.out_channels, ksize, stride=stride, padding=pad
            )
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(nn.Module):
    def __init__(
            self,
            channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            up=False,
            down=False,
            kernel_size=3,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        padding = 1 if kernel_size == 3 else 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv1d(self.out_channels, self.out_channels, kernel_size, padding=padding)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv1d(
                channels, self.out_channels, kernel_size, padding=padding
            )
        else:
            self.skip_connection = nn.Conv1d(channels, self.out_channels, 1)

    def forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AudioMiniEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 base_channels=128,
                 depth=2,
                 resnet_blocks=2,
                 attn_blocks=4,
                 num_attn_heads=4,
                 dropout=0,
                 downsample_factor=2,
                 kernel_size=3):
        super().__init__()
        self.init = nn.Sequential(
            nn.Conv1d(spec_dim, base_channels, 3, padding=1)
        )
        ch = base_channels
        res = []
        for l in range(depth):
            for r in range(resnet_blocks):
                res.append(ResBlock(ch, dropout, kernel_size=kernel_size))
            res.append(Downsample(ch, use_conv=True, out_channels=ch*2, factor=downsample_factor))
            ch *= 2
        self.res = nn.Sequential(*res)
        self.final = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            nn.Conv1d(ch, embedding_dim, 1)
        )
        attn = []
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads,))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim

    def forward(self, x):
        h = self.init(x)
        h = self.res(h)
        h = self.final(h)
        h = self.attn(h)
        return h[:, :, 0]


DEFAULT_MEL_NORM_FILE = os.path.join(os.path.dirname(os.path.realpath('/content/linus-original-DEMO.mp3')), '../data/mel_norms.pth')


class TorchMelSpectrogram(nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, mel_fmin=0, mel_fmax=8000,
                 sampling_rate=22050, normalize=False, mel_norm_file=DEFAULT_MEL_NORM_FILE):
        super().__init__()
        # These are the default tacotron values for the MEL spectrogram.
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate
        self.mel_stft = torchaudio.transforms.MelSpectrogram(n_fft=self.filter_length, hop_length=self.hop_length,
                                                             win_length=self.win_length, power=2, normalized=normalize,
                                                             sample_rate=self.sampling_rate, f_min=self.mel_fmin,
                                                             f_max=self.mel_fmax, n_mels=self.n_mel_channels,
                                                             norm="slaney")
        self.mel_norm_file = mel_norm_file
        if self.mel_norm_file is not None:
            self.mel_norms = torch.load(self.mel_norm_file)
        else:
            self.mel_norms = None

    def forward(self, inp):
        if len(inp.shape) == 3:  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            inp = inp.squeeze(1)
        assert len(inp.shape) == 2
        if torch.backends.mps.is_available():
            inp = inp.to('cpu')
        self.mel_stft = self.mel_stft.to(inp.device)
        mel = self.mel_stft(inp)
        # Perform dynamic range compression
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if self.mel_norms is not None:
            self.mel_norms = self.mel_norms.to(mel.device)
            mel = mel / self.mel_norms.unsqueeze(0).unsqueeze(-1)
        return mel


class CheckpointedLayer(nn.Module):
    """
    Wraps a module. When forward() is called, passes kwargs that require_grad through torch.checkpoint() and bypasses
    checkpoint for all other args.
    """
    def __init__(self, wrap):
        super().__init__()
        self.wrap = wrap

    def forward(self, x, *args, **kwargs):
        for k, v in kwargs.items():
            assert not (isinstance(v, torch.Tensor) and v.requires_grad)  # This would screw up checkpointing.
        partial = functools.partial(self.wrap, **kwargs)
        return partial(x, *args)

import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        kernel_size=3,
        do_checkpoint=True,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.do_checkpoint = do_checkpoint
        padding = 1 if kernel_size == 3 else 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv1d(self.out_channels, self.out_channels, kernel_size, padding=padding)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv1d(
                dims, channels, self.out_channels, kernel_size, padding=padding
            )
        else:
            self.skip_connection = nn.Conv1d(dims, channels, self.out_channels, 1)

    def forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AudioMiniEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 base_channels=128,
                 depth=2,
                 resnet_blocks=2,
                 attn_blocks=4,
                 num_attn_heads=4,
                 dropout=0,
                 downsample_factor=2,
                 kernel_size=3):
        super().__init__()
        self.init = nn.Sequential(
            nn.Conv1d(spec_dim, base_channels, 3, padding=1)
        )
        ch = base_channels
        res = []
        self.layers = depth
        for l in range(depth):
            for r in range(resnet_blocks):
                res.append(ResBlock(ch, dropout, do_checkpoint=False, kernel_size=kernel_size))
            res.append(Downsample(ch, use_conv=True, out_channels=ch*2, factor=downsample_factor))
            ch *= 2
        self.res = nn.Sequential(*res)
        self.final = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            nn.Conv1d(ch, embedding_dim, 1)
        )
        attn = []
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads, do_checkpoint=False))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim

    def forward(self, x):
        h = self.init(x)
        h = self.res(h)
        h = self.final(h)
        for blk in self.attn:
            h = blk(h)
        return h[:, :, 0]


class AudioMiniEncoderWithClassifierHead(nn.Module):
    def __init__(self, classes, distribute_zero_label=True, **kwargs):
        super().__init__()
        self.enc = AudioMiniEncoder(**kwargs)
        self.head = nn.Linear(self.enc.dim, classes)
        self.num_classes = classes
        self.distribute_zero_label = distribute_zero_label

    def forward(self, x, labels=None):
        h = self.enc(x)
        logits = self.head(h)
        if labels is None:
            return logits
        else:
            if self.distribute_zero_label:
                oh_labels = nn.functional.one_hot(labels, num_classes=self.num_classes)
                zeros_indices = (labels == 0).unsqueeze(-1)
                # Distribute 20% of the probability mass on all classes when zero is specified, to compensate for dataset noise.
                zero_extra_mass = torch.full_like(oh_labels, dtype=torch.float, fill_value=.2/(self.num_classes-1))
                zero_extra_mass[:, 0] = -.2
                zero_extra_mass = zero_extra_mass * zeros_indices
                oh_labels = oh_labels + zero_extra_mass
            else:
                oh_labels = labels
            loss = nn.functional.cross_entropy(logits, oh_labels)
            return loss

import os
from glob import glob
import io
import librosa
#import plotly.express as px
import torch
import torch.nn.functional as F
import torchaudio
import numpy as numpy
from scipy.io.wavfile import read


# load_audio


def load_audio(audiopath, sampling_rate=22000):
    if isinstance(audiopath, str):
        if audiopath.endswith('.mp3'):
            audio, lsr = librosa.load(audiopath, sr=sampling_rate)
            audio = torch.tensor(audio, dtype=torch.float32)
        else:
            assert False, f"Unsupported audio format provided:{audiopath[-4:]}"
    elif isinstance(audiopath, io.BytesIO):
        audio, lsr = torchaudio.load(audiopath)
        audio = audio[0]
    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
    if torch.any(audio > 2) or not torch.any(audio < 0):
        print(f"Error with audio data. Max={audio.max()} min={audio.min()}")
    audio.clip_(-1, 1)
    return audio.unsqueeze(0)


# classifier
def classify_audio_clip(clip):
    """
    Returns whether or not Tortoises' classifier thinks the given clip came from Tortoise.
    :param clip: torch tensor containing audio waveform data (get it from load_audio)
    :return: True if the clip was classified as coming from Tortoise and false if it was classified as real.
    """
    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4,
                                                    resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32,
                                                    dropout=0, kernel_size=5, distribute_zero_label=False)
    state_dict = torch.load('classifier.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(state_dict)
    clip = clip.cpu().unsqueeze(0)
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0]

# import os
# import torchaudio
# import numpy as np
# # Load an audio file as a PyTorch tensor
# audiopath = 'song.mp3'
# audio_tensor = load_audio(audiopath)
# # Classify the audio clip
# classification_result = classify_audio_clip(audio_tensor)
# print(f'The Given audio is {classification_result * 100:.2f}% likely to be AI Generated.')





def main():
    st.set_page_config(layout="wide")

    st.title("Audio Forgery Alert:Uncovering Artificial audio with DeepFake Detection")
    # file uploader
    uploaded_file = st.file_uploader(
        "Insert the audio file ", type=['mp3', 'mp4'])
    if uploaded_file is not None:
        if st.button("Check the Audio"):
            col1, col2 = st.columns(2)

            with col1:
                # st.info("your results are below")
                # load andclassify the audio file
                audio_clip = load_audio(uploaded_file)
                result = classify_audio_clip(audio_clip)
                result = result.item()
                st.info(f"Result Probability: {result * 100}")
                st.success(
                    f"The uploaded audio is {result * 100:.2f}% likely to be AI Generated.")

            with col2:
                st.info("Your Uploaded audio is below")
                st.audio(uploaded_file)


if __name__ == '__main__':
    main()


