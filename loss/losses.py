import torch

STYLE_WEIGHTS = [1.0, 0.8, 0.5, 0.3, 0.1]

def get_content_loss(base_content : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
    return torch.mean((base_content - target) ** 2)

def gram_matrix(x : torch.Tensor) -> torch.Tensor:
  channels = x.size(dim=0)
  g = x.view(channels, -1)
  return torch.mm(g, g.mT) / g.size(dim=1)
  
def get_style_loss(base_style, gram_target):
    loss = 0
    for i, (base, target) in enumerate(zip(base_style, gram_target)):
        gram_style = gram_matrix(base)
        loss += STYLE_WEIGHTS[i] * torch.mean((gram_style - target) ** 2)
    return loss

