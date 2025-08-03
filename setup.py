from setuptools import setup, find_packages

setup(
    name='style_transfer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'pillow',
        'numpy',
        'tqdm'
    ],
    entry_points={
        'console_scripts': [
            'style_transfer=main:main',
        ],
    },
    author='mishcum',
    description='Neural Style Transfer with PyTorch and VGG19',
    python_requires='>=3.8',
)