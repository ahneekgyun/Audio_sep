import numpy as np
from typing import Dict, List, NoReturn, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import STFT, ISTFT, magphase
from models.base import Base, init_layer, init_bn, act
from lavt_lib.dev.segmentation_dev import lavt

class LAAT_base(nn.Module, Base):
    def __init__(self, input_channels, output_channels):
        super(LAAT_base, self).__init__()

        window_size = 1024
        hop_size = 160
        center = True
        pad_mode = "reflect"
        window = "hann"
        momentum = 0.01

        self.output_channels = output_channels
        self.target_sources_num = 1
        self.K = 3
        
        self.time_downsample_ratio = 2 ** 5  # This number equals 2^{#encoder_blcoks}

        # lavt 모델 불러오기
        self.lavt_model = lavt(pretrained='', args=None)

        # CLAP을 768로 변환하기 위한 선형 투영
        self.projection_clap = nn.Linear(512,768)

        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum)

        self.pre_conv = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=32, 
            kernel_size=(1, 1), 
            stride=(1, 1), 
            padding=(0, 0), 
            bias=True,
        )

        self.after_conv = nn.Conv2d(
            in_channels=32,
            out_channels=output_channels * self.K,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.pre_conv)
        init_layer(self.after_conv)

    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
    ) -> torch.Tensor:
        r"""Convert feature maps to waveform.

        Args:
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform


    def forward(self, mixtures, conditions):
        """
        Args:
          input: (batch_size, segment_samples, channels_num)

        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """

        mag, cos_in, sin_in = self.wav_to_spectrogram_phase(mixtures)
        x = mag

        # freq_bins 기준으로 Batch normalization
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""
        batch_size = x.shape[0]
        
        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio)) * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, channels, T, F)
        # torch.Size([12, 1, 1024, 512])

        # 3채널로 확장, 스펙토그램 -> 이미지 처리용
        # expand는 뷰객체 만드는 방법
        x = x.expand(-1, 3, -1, -1)
        # repeat는 메모리 복사
        # x = x.repeat(1,3,1,1)
        # torch.Size([12, 3, 1024, 512])

        # Lavt 적용한 부분

        # condition 처리 CLAP
        l_mask = torch.ones(batch_size, 1, 1).to(x.device)
        conditions = self.projection_clap(conditions)
        conditions = conditions.unsqueeze(-1)
        
        # 스펙토그램과 이미지 달라서 축 변환
        x = x.permute(0, 1, 3, 2)

        x = self.lavt_model(x,conditions,l_mask)
        x = x.permute(0, 1, 3, 2)
        # x : 32 channel output

        x = self.after_conv(x)
        # output channel : output_channels * self.K

        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]

        audio_length = mixtures.shape[2]

        # Recover each subband spectrograms to subband waveforms. Then synthesis
        # the subband waveforms to a waveform.
        separated_audio = self.feature_maps_to_wav(
            input_tensor=x,
            # input_tensor: (batch_size, target_sources_num * output_channels * self.K, T, F')
            sp=mag,
            # sp: (batch_size, input_channels, T, F')
            sin_in=sin_in,
            # sin_in: (batch_size, input_channels, T, F')
            cos_in=cos_in,
            # cos_in: (batch_size, input_channels, T, F')
            audio_length=audio_length,
        )
        # （batch_size, target_sources_num * output_channels, subbands_num, segment_samples)

        output_dict = {'waveform': separated_audio}

        return output_dict




class LAAT(nn.Module):
    def __init__(self, input_channels, output_channels, condition_size):
        super(LAAT, self).__init__()

        self.base = LAAT_base(
            input_channels=input_channels, 
            output_channels=output_channels,
        )


    def forward(self, input_dict):
        
        mixtures = input_dict['mixture']
        conditions = input_dict['condition']

        output_dict = self.base(
            mixtures=mixtures, 
            conditions=conditions,
        )

        return output_dict

    @torch.no_grad()
    def chunk_inference(self, input_dict):
        chunk_config = {
                    'NL': 1.0,
                    'NC': 3.0,
                    'NR': 1.0,
                    'RATE': 32000
                }

        mixtures = input_dict['mixture']
        conditions = input_dict['condition']

        film_dict = self.film(
            conditions=conditions,
        )

        NL = int(chunk_config['NL'] * chunk_config['RATE'])
        NC = int(chunk_config['NC'] * chunk_config['RATE'])
        NR = int(chunk_config['NR'] * chunk_config['RATE'])

        L = mixtures.shape[2]
        
        out_np = np.zeros([1, L])

        WINDOW = NL + NC + NR
        current_idx = 0

        while current_idx + WINDOW < L:
            chunk_in = mixtures[:, :, current_idx:current_idx + WINDOW]

            chunk_out = self.base(
                mixtures=chunk_in, 
                film_dict=film_dict,
            )['waveform']
            
            chunk_out_np = chunk_out.squeeze(0).cpu().data.numpy()

            if current_idx == 0:
                out_np[:, current_idx:current_idx+WINDOW-NR] = \
                    chunk_out_np[:, :-NR] if NR != 0 else chunk_out_np
            else:
                out_np[:, current_idx+NL:current_idx+WINDOW-NR] = \
                    chunk_out_np[:, NL:-NR] if NR != 0 else chunk_out_np[:, NL:]

            current_idx += NC

            if current_idx < L:
                chunk_in = mixtures[:, :, current_idx:current_idx + WINDOW]
                chunk_out = self.base(
                    mixtures=chunk_in, 
                    film_dict=film_dict,
                )['waveform']

                chunk_out_np = chunk_out.squeeze(0).cpu().data.numpy()

                seg_len = chunk_out_np.shape[1]
                out_np[:, current_idx + NL:current_idx + seg_len] = \
                    chunk_out_np[:, NL:]

        return out_np


