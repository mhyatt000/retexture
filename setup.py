
from setuptools import setup, find_packages

setup(
    name='retexture',
    version='0.1.0',
    description='using blender to augment 3d model textures',
    author='Matt Hyatt',
    author_email='mhyatt@luc.edu',
    url='https://github.com/mhyatt000/retexture',
    packages=find_packages(),
    install_requires=[
        'hydra-core',
    ],
    entry_points={
        'console_scripts': [
            'retexture = retexture.main:_main',
        ],
    },
    classifiers=[],
)
