import random
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# block module that packages linear, bn, relu and dropout layers
class Block(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(in_features=in_features, out_features=out_features, bias=False),
                                  nn.ReLU(),
                                  nn.Dropout(dropout))
        
    def forward(self, inputs):
        return self.layers(inputs)


# Baseline model
class RelationClassifierPro(nn.Module):
    def __init__(self, in_features, out_features,dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
                                    Block(in_features, 256, dropout=dropout),
                                    Block(256, 64, dropout=dropout),
                                    Block(64, 64, dropout=dropout),
                                    nn.Linear(64, out_features)
                                    )
    
    def forward(self, inputs):  # x (batch_size, input_dim)
        return self.layers(inputs)
    

# Final LSTM model
class SequenceTagger(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=False, dropout=0.3):
        super(SequenceTagger, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # h0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # out, _ = self.lstm(x, (h0, c0))
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.fc(out[:, :, :])
        return out