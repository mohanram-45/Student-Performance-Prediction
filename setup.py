from setuptools import find_packages, setup
from typing import List
from pathlib import Path

Hyphen_e_dot = "-e ."

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(Path(__file__).parent / file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if Hyphen_e_dot in requirements:
             requirements.remove(Hyphen_e_dot)
             
    return requirements



setup(
    name = "student-performance",
    version="0.0.1",
    author = "Mohan",
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt")
)