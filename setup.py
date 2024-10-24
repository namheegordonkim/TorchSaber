# Copyright 2024 Nam Hee Gordon Kim.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""setup.py for TorchSaber.

Install for development:

  pip intall -e .
"""

from setuptools import find_packages
from setuptools import setup

setup(
    name="torch-saber",
    version="0.0.1",
    description="GPU-enabled XROR visualization and collision processing using PyTorch and PyVista",
    author="Nam Hee Gordon Kim",
    author_email="namhee.kim@aalto.fi",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="http://namheegordonkim.github.io",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "etils",
        "fpzip",
        "matplotlib",
        "mujoco",
        "pyvista",
        "pyvista-imgui",
        "pymongo",
        "torch",
        "tqdm",
        "scipy",
        "imgui-bundle",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="BeatSaber XROR PyTorch PyVista collision",
)
