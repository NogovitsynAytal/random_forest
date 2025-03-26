from setuptools import setup, find_packages

setup(
    name="random_forest",
    version="0.1.0",
    description="A simple RandomForest implementation in Python",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)