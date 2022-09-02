from sys import stderr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math,copy

class MLP(nn.Module):
    def __init__(self, units: list, hidden_layer_activation=nn.ReLU(), drop_out_prob=[0.2, 0.5], optional_last_layer=None, batch_normalization=False):
        super(MLP, self).__init__()
        self.modules = []
        self.log_counter = 0

        for level in range(len(units)-2):
            self.modules.append(nn.Dropout(p=drop_out_prob[level]))
            self.modules.append(nn.Linear(units[level], units[level+1]))
            if(hidden_layer_activation!= None):self.modules.append(hidden_layer_activation)
            if(batch_normalization):self.modules.append(nn.BatchNorm1d(units[level+1]))

        self.modules.append(nn.Linear(units[-2], units[-1]))

        if optional_last_layer != None:
            self.modules.append(optional_last_layer)

        self.mlp = nn.Sequential(*self.modules)

        for layer in self.modules:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                # torch.nn.init.xavier_uniform_(layer.bias)
                layer.bias.data.fill_(0.01)            
    def forward(self, X):
        for layer in self.modules: X = layer(X)
        return X
    def draw_layers(self,summary_writer,name):
        images = []
        mean_layer_weight = []
        layer_bias = []
        
        for layer in self.modules:
            if isinstance(layer, nn.Linear):
                mean_layer = torch.mean(layer.state_dict()['weight'],axis=0)
                mean_layer_weight += [mean_layer]
                layer_bias += [layer.state_dict()['bias']]
                images += [mean_layer.reshape(int(math.sqrt(int(mean_layer.shape[0]))),-1).detach().numpy()]
        
        for index,layer in enumerate(images):
            summary_writer.add_image('{}_layers/layer{}:time{}'.format(name,index,self.log_counter), layer[None, :])
        
        for index,weight in enumerate(mean_layer_weight):
            try:
                summary_writer.add_histogram("weights/{}_layers/layer{}".format(name,index),weight.view(-1), global_step=self.log_counter)
            except Exception as exp:
                print(exp)

        for index,bias in enumerate(layer_bias):
            try:
                summary_writer.add_histogram("biases/{}_layers/layer{}".format(name,index),bias.view(-1), global_step=self.log_counter)
            except Exception as exp:
                print(exp)

        self.log_counter = (self.log_counter+1)%1000

        
class StochActor(nn.Module):
    """This stochastic actor learns a Gaussian distribution. While the mean value 
    is learnt by a full-fledged MLP, the standard deviation is denoted by a
    single trainable weight. With Pytorch's distribution package, probabilities 
    given a certain distribution and an action can be calculated."""
    def __init__(self,m_net_units,m_net_hidden_layer_activation,m_net_drop_out_prob,m_net_normalize,m_net_optional_last_layer,s_net_units,s_net_hidden_layer_activation,s_net_drop_out_prob,s_net_actor_normalize,s_net_optional_last_layer):
        super().__init__()
        self.mu_net = MLP(units=m_net_units, hidden_layer_activation=m_net_hidden_layer_activation, drop_out_prob=m_net_drop_out_prob,batch_normalization=m_net_normalize,optional_last_layer=m_net_optional_last_layer)
        self.sigma_net = MLP(units=s_net_units, hidden_layer_activation=s_net_hidden_layer_activation, drop_out_prob=s_net_drop_out_prob,batch_normalization=s_net_actor_normalize,optional_last_layer=s_net_optional_last_layer)
    def _distribution(self, state):
        try:
            Ɛ = 0.0000001
            return torch.distributions.normal.Normal(loc=self.mu_net(state)+Ɛ, scale=self.sigma_net(state)+Ɛ)
        except Exception as exp:
            print('######## {}'.format(exp))
            print('state = {}'.format(state))
            print('-----------------------')
            print('sigma(state) = {}'.format(self.mu_net(state)))
            print('mu(state) = {}'.format(self.sigma_net(state)))
            print('======================')
    def forward(self, state, action=None):
        pi = self._distribution(state)
        # if action is None, logp_a will be, too
        if action is None:
            logp_a = None
        else:
            logp_a = pi.log_prob(action).sum(axis=-1)
        return pi, logp_a