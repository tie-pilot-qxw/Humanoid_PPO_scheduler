# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch import nn

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self,actions, states, in_ch=1, out_ch=1):
        super(U_Net, self).__init__()
        self.actions = actions
        self.states = states
        self.activation = nn.LeakyReLU()
        self.linear1 = nn.Linear(self.states, 128)
        self.transformer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=256)
        self.linear2 = nn.Linear(128, actions)
        # self.linear1 = nn.Linear(self.states, 512)
        # self.linear2 = nn.Linear(512, 256)
        # self.linear3 = nn.Linear(256, 512)
        # self.linear4 = nn.Linear(512, 256)
        # self.linear5 = nn.Linear(256, 512)
        # self.linear6 = nn.Linear(512, self.actions)
        # self.start_dim = 128

        # self.upscale = nn.Linear(self.states, self.start_dim)

        # n1 = 4
        # filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        # self.Maxpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        # self.Maxpool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        # self.Maxpool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        # self.Maxpool4 = nn.AvgPool1d(kernel_size=2, stride=2)

        # self.Conv1 = conv_block(in_ch, filters[0])
        # self.Conv2 = conv_block(filters[0], filters[1])
        # self.Conv3 = conv_block(filters[1], filters[2])
        # self.Conv4 = conv_block(filters[2], filters[3])
        # self.Conv5 = conv_block(filters[3], filters[4])

        # self.Up5 = up_conv(filters[4], filters[3])
        # self.Up_conv5 = conv_block(filters[4], filters[3])

        # self.Up4 = up_conv(filters[3], filters[2])
        # self.Up_conv4 = conv_block(filters[3], filters[2])

        # self.Up3 = up_conv(filters[2], filters[1])
        # self.Up_conv3 = conv_block(filters[2], filters[1])

        # self.Up2 = up_conv(filters[1], filters[0])
        # self.Up_conv2 = conv_block(filters[1], filters[0])

        # self.Conv = nn.Conv1d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        # self.cls = nn.Linear(self.start_dim, self.actions)
       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        x = self.linear1(x)

        x = self.activation(x)

        x = self.transformer(x)

        out = self.linear2(x)
        
        # l1 = self.linear1(x)

        # l1 = self.activation(l1)

        # l2 = self.linear2(l1)

        # l2 = self.activation(l2)

        # l2 = self.linear3(l2)

        # l2 = self.activation(l2) + l1

        # l3 = self.linear4(l2)

        # l3 = self.activation(l3)

        # l3 = self.linear5(l3)

        # l3 = self.activation(l3) + l2

        # out = self.linear6(l3)

        # x = self.upscale(x)
        # x = x.unsqueeze(1)

        # e1 = self.Conv1(x)
        
        # e2 = self.Maxpool1(e1)
        # e2 = self.Conv2(e2)

        # e3 = self.Maxpool2(e2)
        # e3 = self.Conv3(e3)

        # e4 = self.Maxpool3(e3)
        # e4 = self.Conv4(e4)

        # e5 = self.Maxpool4(e4)
        # e5 = self.Conv5(e5)

        # d5 = self.Up5(e5)
        # d5 = torch.cat((e4, d5), dim=1)

        # d5 = self.Up_conv5(d5)

        # d4 = self.Up4(d5)
        # d4 = torch.cat((e3, d4), dim=1)
        # d4 = self.Up_conv4(d4)

        # d3 = self.Up3(d4)
        # d3 = torch.cat((e2, d3), dim=1)
        # d3 = self.Up_conv3(d3)

        # d2 = self.Up2(d3)
        # d2 = torch.cat((e1, d2), dim=1)
        # d2 = self.Up_conv2(d2)

        # out = self.Conv(d2)
        # out = out.squeeze(1)

        # out = self.cls(out)

        return out

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        #self.actor = U_Net(num_actions, num_actor_obs)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
