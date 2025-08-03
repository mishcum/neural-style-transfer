from utils.image_utils import get_images
from style_transfer import style_transfer

def main():
    img_path = 'img.jpg'
    img_style_path = 'img_style.jpg'

    img, img_style, result_img = get_images(img_path, img_style_path)
    image = style_transfer(img, img_style, result_img)
    image.save("result.jpg")
    

if __name__ == '__main__':
    main()