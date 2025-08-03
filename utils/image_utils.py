from PIL import Image
import torch
from torchvision.transforms import v2
from typing import Tuple

def get_images(img_path : str, style_img_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    img = Image.open('img.jpg').convert('RGB')
    img_style = Image.open('img_style.jpg').convert('RGB')
    
    img = transforms(img).unsqueeze(0)
    result_img = img.clone()
    result_img.requires_grad_()
    img_style = transforms(img_style).unsqueeze(0)

    return img, img_style, result_img
    

transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])