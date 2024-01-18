import itertools
import json
import os
import os.path as osp
import time
from concurrent.futures import ProcessPoolExecutor

from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = osp.dirname(osp.dirname(__file__))

base_config = osp.join(ROOT, "configs", "base.yaml")
other = osp.join(ROOT, "configs", "humanredo.yaml")
cfg = OmegaConf.load(other)


def load_data(cfg):
    """loads file paths for models and textures"""
    model_dir = osp.join(ROOT, cfg.data_dir, "models")
    tex_dir = osp.join(ROOT, cfg.data_dir, "textures")

    get_full_paths = lambda p: [osp.join(p, c) for c in os.listdir(p)]
    get_filtered = (
        lambda a, b: [
            y for y in get_full_paths(a) if any([basename(y).split(".")[0] == x for x in b])
        ]
        if b
        else get_full_paths(a)
    )

    models = get_filtered(model_dir, cfg.get("models"))
    textures = get_filtered(tex_dir, cfg.get("textures"))

    return models, textures


def cfg2cmd(config):
    """Converts an OmegaConf configuration to a string of command-line arguments."""

    args = []
    keys = [
        "data_dir",
        "out_dir",
        "nangles",
        "file_type",
        "quiet",
        "model",
        "texture",
    ]

    for key, value in config.items():
        if key not in keys:
            continue

        # For boolean flags, add only the key if the value is True
        if isinstance( value, bool):  
            if value:
                args.append(f"--{key}")
        else:  
            args.append(f"--{key} {str(value)}")

    return " ".join(args)


def basename(file):
    """return filename with no extensions"""
    return osp.basename(file).split(".")[0]


def render(pair):
    """renders a model & texture pair"""

    model, texture = pair
    quiet = "> /dev/null 2>&1" if cfg.quiet else ""
    blend_script = osp.join(ROOT, "retexture", "scripts", "blend.py")
    blend = f"blender -b -P {blend_script} -- {cfg2cmd(cfg)}"
    model_args = " " + cfg2cmd({"model": model, "texture": texture})
    os.system(blend + model_args + quiet)


def pair_complete(pair):
    """skip the ones that have been made"""

    model, texture = pair
    out_pair = osp.join(ROOT, cfg.out_dir, basename(model), basename(texture))
    if osp.exists(out_pair) and len(os.listdir(out_pair)) >= cfg.nangles:
        return True
    return False


def filter_pairs(pair):
    """filter pairs by model or texture"""

    model = basename(pair[0]).split('.')[0]
    texture = basename(pair[1]).split('.')[0]

    if any(model in p[0] and texture in p[1] for p in cfg.pairs):
        return True
    return False

def main():
    print(cfg)

    # shell commands
    media = "python ~/cs/retexture/retexture/util/walk_media.py"

    models, textures = load_data(cfg)
    pairs = list(itertools.product(models, textures))
    pairs = [p for p in pairs if not pair_complete(p)]

    if cfg.get('pairs'):
        pairs = list(filter(filter_pairs, pairs))

    with ProcessPoolExecutor() as ex:
        desc = "Rendering 3D Models..."
        for _ in tqdm(ex.map(render, pairs), total=len(pairs), desc=desc):
            pass

    # os.system(media)


if __name__ == "__main__":
    main()
