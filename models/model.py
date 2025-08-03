import torch
from torch import nn
from torchvision import models

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mf = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.mf.requires_grad_(False)
        self.mf.eval()

        self.out_indices = [0, 5, 10, 19, 28, 34]
        self.style_layers = len(self.out_indices) - 1 

    def forward(self, x):
        outputs = []
        for indx, layer in enumerate(self.mf):
            x = layer(x)
            if indx in self.out_indices:
                outputs.append(x.squeeze(0))
        return outputs
