import os

from setuptools import find_packages
from setuptools import setup

setup(
    name='lomap',
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    python_requires='>=3.9',
    install_requires=[
        # === d4rl related ===
        "gym==0.23.1",
        'numpy<1.23.0',
        "mujoco_py==2.1.2.14",
        "mujoco==3.1.6",
        "pybullet",
        "h5py",
        "termcolor",  # adept_envs dependency
        "click",  # adept_envs dependency
        "dm_control==1.0.20",
        "mjrl @ git+https://github.com/aravindr93/mjrl@master#egg=mjrl",
        "cython<3",
        # ======================
        --extra-index-url https://download.pytorch.org/whl/cu113
        torch==1.11.0+cu113
        'matplotlib<=3.7.5',
        'hydra-core',
        'einops',
        'faiss-gpu',
        'numba<0.60.0',
        'zarr<2.17',
    ]
)

