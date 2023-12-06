import itertools
import json
import os
import os.path as osp
import time
from concurrent.futures import ProcessPoolExecutor

from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = osp.dirname(__file__)

config_path = osp.join(ROOT, "configs", "base.json")
cfg = OmegaConf.load(config_path)
print(cfg)


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


def basename(file):
    """return filename with no extensions"""
    return osp.basename(file).split(".")[0]


def render(pair):
    """renders a model & texture pair"""

    model, texture = pair
    quiet = " > /dev/null 2>&1"
    blend = f"blender -b -P ~/cs/retexture/retexture/main.py -- {cfg2cmd(cfg)}"
    model_args = " " + cfg2cmd({"model": model, "texture": texture})
    os.system(blend + model_args + quiet)


def pair_complete(pair):
    """skip the ones that have been made"""

    model, texture = pair
    out_pair = osp.join(ROOT, cfg.out_dir, basename(model), basename(texture))
    if osp.exists(out_pair) and len(os.listdir(out_pair)) >= cfg.nangles:
            return True
    return False


def main():

    # shell commands
    quiet = " > /dev/null 2>&1"
    media = "python ~/cs/retexture/retexture/util/walk_media.py"

    # if input(f"delete {cfg.out_dir}? (y/n) ") == "y":
        # print("deleting {cfg.out_dir}...")
        # os.system(f"rm -r {cfg.out_dir} + {quiet}")

    models, textures = load_data(cfg)
    pairs = list(itertools.product(models, textures))
    pairs = [p for p in pairs if  not pair_complete(p) ] 

    with ProcessPoolExecutor() as ex:
        desc = "Rendering 3D Models..."
        for _ in tqdm( ex.map(render, pairs), total=len(pairs), desc=desc):
            pass

    # os.system(media)


if __name__ == "__main__":
    main()
