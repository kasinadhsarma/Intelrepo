from setuptools import setup, find_packages

setup(
    name="backend",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'opencv-python>=4.8.1.78',
        'numpy>=1.22.0',
        'torch>=2.2.0',
        'torchvision>=0.10.0',
        'pillow>=10.0.1',
        'fastapi>=0.109.1',
        'uvicorn>=0.27.0',
        'python-multipart>=0.0.9',
        'langchain>=0.1.5',
        'langchain-community>=0.0.16',
    ],
)