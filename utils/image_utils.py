from PIL import Image
import torch
from torchvision.transforms import v2
from typing import Tuple

TRANSFORMS = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

def get_images(img_path : str, style_img_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    img = Image.open(img_path).convert('RGB')
    img_style = Image.open(style_img_path).convert('RGB')

    img_style = img_style.resize(img.size, Image.LANCZOS)
    
    img = TRANSFORMS(img).unsqueeze(0)
    result_img = img.clone()
    result_img.requires_grad_()
    img_style = TRANSFORMS(img_style).unsqueeze(0)

    return img, img_style, result_img
    
def postprocess(tensor : torch.Tensor) -> Image.Image:
    x = tensor.detach().squeeze()
    low, hi = torch.amin(x), torch.amax(x)
    x = (x - low) / (hi - low) * 255.0
    x = x.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(x)

