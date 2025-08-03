import argparse
import os
from utils.image_utils import get_images
from style_transfer import style_transfer
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Neural Style Transfer using VGG19')

    parser.add_argument('--content', type=str, required=True, help='Path to content image')
    parser.add_argument('--style', type=str, required=True, help='Path to style image')
    parser.add_argument('--output', type=str, default=None, help='Output image path (optional)')

    parser.add_argument('--epochs', type=int, default=100, help='Number of optimization steps')
    parser.add_argument('--style-weight', type=float, default=1000.0, help='Weight for style loss')
    parser.add_argument('--content-weight', type=float, default=1.0, help='Weight for content loss')

    return parser.parse_args()

def generate_output_path(content_path: str) -> str:
    dir_name, file_name = os.path.split(content_path)
    name, ext = os.path.splitext(file_name)
    base_name = f'{name}_stylized'
    candidate = os.path.join(dir_name, base_name + ext)
    
    index = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dir_name, f'{base_name}_{index}{ext}')
        index += 1

    return candidate

def main():
    args = parse_args()

    set_device()

    img, img_style, result_img = get_images(args.content, args.style)
    result_image = style_transfer(
        img,
        img_style,
        result_img,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        epochs=args.epochs
    )

    output_path = args.output if args.output else generate_output_path(args.content)
    result_image.save(output_path)

def set_device():
    if torch.mps.is_available():
        torch.set_default_device('mps')
    elif torch.cuda.is_available():
        torch.set_default_device('cuda')
    else:
        torch.set_default_device('cpu')

if __name__ == '__main__':
    main()
