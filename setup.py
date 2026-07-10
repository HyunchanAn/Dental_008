from setuptools import setup, find_packages

setup(
    name="dentex_seg",
    version="0.1.0",
    description="DENTEX Mixed Dentition Instance Segmentation",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "datasets>=2.14.0",
        "opencv-python>=4.8.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "tqdm>=4.65.0"
    ],
)
