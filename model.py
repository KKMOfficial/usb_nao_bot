import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, units: list, hidden_layer_activation='relu', init_type='default', drop_out_prob=0.2):
        super(MLP, self).__init__()
        self.units = units
        self.init_type = init_type
        self.n_layers = len(units) # including input and output layers
        valid_activations = {'relu': nn.ReLU(),
                             'tanh': nn.Tanh(),
                             'sigmoid': nn.Sigmoid(),
                             'null': None}
        self.modules = []
        self.activation = valid_activations[hidden_layer_activation]
        for level in range(len(units)-2):
          self.modules.append(nn.Linear(units[level], units[level+1]))
          if(self.activation!= None):self.modules.append(self.activation)
          self.modules.append(nn.Dropout(p=drop_out_prob))
        self.modules.append(nn.Linear(units[-2], units[-1]))

        self.mlp = nn.Sequential(*self.modules)

        def init_weight(layer):
          if isinstance(layer, nn.Linear):
            if self.init_type=='uniform':
              layer.weight.data.uniform_(0.0, 1.0),
              layer.bias.data.uniform_(0.0, 1.0),
            elif self.init_type=='normal':
              layer.weight.data.normal_(mean=0.0, std=1.0)
              layer.bias.data.normal_(mean=0.0, std=1.0)
            elif self.init_type=='zero':
              layer.weight.data.fill_(0)
              layer.bias.data.fill_(0)
            elif self.init_type=='default':
              pass
        
        self.mlp.apply(init_weight)

    def forward(self, X):
        for layer in self.modules:
          X = layer(X)
        return X
        