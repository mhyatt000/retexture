# retexture
using blender to augment 3d model textures

<img src="assets/teaser.png" alt="" width="100%"/>

## Installation

1. clone the repo (more instructions later)
2. `pip install -e .`

### Installing blender

* MacOS: `brew install blender`
* Unix: `apt-get install blender`

### Install Blender via script

see `./install.sh`

## Dataset Setup

place all models in `datasets/models` and place textures in `datasets/textures` like so:

```
datasets
├── models
│   ├── bird1.dae
│   ├── butterfly1.dae
│   ...
├── datasets/textures
│   ├── bird1.jpg
│   ├── butterfly1.png
│   ...
```

## Config Setup

see `configs/base_config.yaml`

## Run

`python -m retexture run`

## NOTES

gpu rendering does not work (easily)
* in the current state, a black image is created
* blender provides lower level gpu support... see [docs](https://docs.blender.org/api/current/gpu.html)

external libraries arediscouraged
* blender uses its own python distro & environment
* used sys.path.insert as a workaround
* custom packages do not work with blender (TBD)

argparse conflicts with blender commandline arguments
* hydra also conflicts
* might be better to nix the configurations or read statically from a config.yaml

memory required increases with job runtime
* maybe blender keeps a history of all operations?
* consider splitting jobs into batches

## Problems

- bounding box centering doesnt work
