import itertools
import json
import os
import os.path as osp
import time

from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = osp.dirname(__file__)


def load_data(cfg):
    """loads file paths for models and textures"""
    model_dir = osp.join(ROOT, cfg.data_dir, "models")
    tex_dir = osp.join(ROOT, cfg.data_dir, "textures")

    get_full_paths = lambda p: [osp.join(p, c) for c in os.listdir(p)]

    models = get_full_paths(model_dir)
    textures = get_full_paths(tex_dir)
    return models, textures


def cfg2cmd(config):
    """Converts an OmegaConf configuration to a string of command-line arguments."""

    args = []
    for key, value in config.items():
        if isinstance(
            value, bool
        ):  # For boolean flags, add only the key if the value is True
            if value:
                args.append(f"--{key}")

        else:  # Use shlex.quote to handle values that need to be quoted (like strings with spaces)
            # args.append(f"--{key} {shlex.quote(str(value))}")
            args.append(f"--{key} {str(value)}")

    return " ".join(args)


def main():
    config_path = osp.join(ROOT, "configs", "base.json")
    cfg = OmegaConf.load(config_path)
    print(cfg)

    # shell commands
    quiet = " > /dev/null 2>&1"
    blend = f"blender -b -P ~/cs/retexture/retexture/main.py -- {cfg2cmd(cfg)}"
    media = "python ~/cs/retexture/retexture/util/walk_media.py"

    os.system(f"rm -r {cfg.out_dir} + {quiet}")

    models, textures = load_data(cfg)
    pairs = list(itertools.product(models, textures))

    for model, texture in tqdm(pairs, desc="Rendering 3D Models..."):
        model_args = ' ' + cfg2cmd({"model": model, "texture": texture})
        os.system(blend + model_args + quiet)

    os.system(media)


if __name__ == "__main__":
    main()
