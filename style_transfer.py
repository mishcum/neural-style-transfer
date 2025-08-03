from models.model import Model
import torch
from torch.optim import Adam
from loss.losses import get_content_loss, gram_matrix, get_style_loss
from tqdm import tqdm
from PIL import Image
import numpy as np


def style_transfer(img : torch.Tensor, img_style : torch.Tensor, result_img : torch.Tensor) -> Image:
    model = Model()

    img_outs = model(img)
    img_style_outs = model(img_style)

    gram_style = [gram_matrix(x) for x in img_style_outs[:model.style_layers]]

    content_weight = 1
    style_weight = 1000
    best_loss = torch.inf

    epochs = 100

    optimizer = Adam(params=[result_img], lr=0.01)
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

    x = best_img.detach().squeeze()
    low, hi = torch.amin(x), torch.amax(x)
    x = (x - low) / (hi - low) * 255.0
    x = x.permute(1, 2, 0)
    x = x.numpy()
    x = np.clip(x, 0, 255).astype('uint8')

    image = Image.fromarray(x, 'RGB')

    return image
