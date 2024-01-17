import numpy as np
import torch
import torch.nn as nn

from transformer.layers import EncoderLayer
from alg_parameters import * 

class Encoder(nn.Module):
    """a encoder model with self attention mechanism"""

    def __init__(self, d_model, d_hidden, n_layers, n_head, d_k, d_v):
        """create multiple computation blocks"""
        super(Encoder, self).__init__()
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_hidden, n_head, d_k, d_v) for _ in range(n_layers)])

    def forward(self, enc_output, return_attns=False):
        """use self attention to merge messages"""
        # enc_output shape:[-1, 8, 512]
        enc_slf_attn_list = []
        enc_out = torch.zeros(enc_output.shape[0], NetParameters.NET_SIZE).to(enc_output.device)
        for i in range(enc_output.shape[0]):
            enc_input = enc_output[i].unsqueeze(0)  # enc_input shape:[1, 8, 512]
            # enc_input = enc_output[i]  # enc_input shape:[8, 512]
            for enc_layer in self.layer_stack:
                enc_input, enc_slf_attn = enc_layer(enc_input)
                enc_slf_attn_list += [enc_slf_attn] if return_attns else []
            enc_input = enc_input.squeeze()
            # print(f"enc_input.shape:{enc_input.shape}")
            agent_index = i % 8
            # print(f"agent_index:{agent_index}")

            enc_out[i] = enc_input[agent_index]
            # print(f"enc_input.squeeze()[i % 8].shape:{enc_input.squeeze()[i % 8].shape}")
            # enc_out[i, :] = enc_input[0, i % 8, :]
        if return_attns:
            return enc_out, enc_slf_attn_list
        # print(f"enc_out.shape:{enc_out.shape}")
        return enc_out


class PositionalEncoding(nn.Module):
    """sinusoidal position embedding"""

    def __init__(self, d_hid, n_position=200):
        """create table"""
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """encode unique agent id """
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class TransformerEncoder(nn.Module):
    """a sequence to sequence model with attention mechanism"""

    def __init__(self, d_model, d_hidden, n_layers, n_head, d_k, d_v, n_position):
        """initialization"""
        super(TransformerEncoder, self).__init__()
        self.encoder = Encoder(d_model=d_model, d_hidden=d_hidden,
                               n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v)

        self.position_enc = PositionalEncoding(d_model, n_position=n_position)

    def forward(self, encoder_input):
        """run encoder"""
        # encoder_input shape [8, 8, 512]
        # print(f"encoder_input.device:{encoder_input.device}")
        encoder_input1 = self.position_enc(encoder_input)
        # print(f"encoder_input1.device:{encoder_input1.device}")
        
        enc_output = self.encoder(encoder_input1)
        # print(f"enc_output.device:{enc_output.device}")

        # print(f"enc_output.shape:{enc_output.shape}")
        return enc_output  # [-1, 512]
