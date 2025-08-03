import torch
from torch.optim import Adam
from tqdm import tqdm

from models.model import Model
from loss.losses import get_content_loss, gram_matrix, get_style_loss
from utils.image_utils import postprocess

def style_transfer(img : torch.Tensor, img_style : torch.Tensor, 
                   result_img : torch.Tensor, content_weight : float = 1.0,
                   style_weight : float = 1000.0, epochs : int = 100, lr : float = 0.01) -> torch.Tensor:
    
    model = Model()
    img_outs = model(img)
    img_style_outs = model(img_style)

    gram_style = [gram_matrix(x) for x in img_style_outs[:model.style_layers]]
    best_loss = torch.inf

    optimizer = Adam(params=[result_img], lr=lr)
    best_img = result_img.clone()

    for _ in tqdm(range(epochs)):
        img_result_outs = model(result_img)

        content_loss = get_content_loss(img_result_outs[-1], img_outs[-1])
        style_loss = get_style_loss(img_result_outs, gram_style)
        loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < best_loss:
            best_loss = loss
            best_img = result_img.clone()
            
    return postprocess(best_img)
