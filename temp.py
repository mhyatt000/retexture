
import itertools
import json
import os
import os.path as osp
import time
from concurrent.futures import ProcessPoolExecutor

from omegaconf import OmegaConf
from tqdm import tqdm


ROOT = osp.dirname((__file__))
print(ROOT)
config_path = osp.join(ROOT, "configs", "base.json")
cfg = OmegaConf.load(config_path)

def load_data(cfg):
    """loads file paths for models and textures"""
    model_dir = osp.join(ROOT, cfg.data_dir, "models")
    tex_dir = osp.join(ROOT, cfg.data_dir, "textures")

    get_full_paths = lambda p: [osp.join(p, c) for c in os.listdir(p)]

    models = get_full_paths(model_dir)
    textures = get_full_paths(tex_dir)
    return models, textures

MOD, TEX = load_data(cfg)

with open('human.txt', 'r') as f:
    sub  = f.read().splitlines()



# sub = [s.split('_')[:-1] for s in sub]
sub = [s.split('.')[0] for s in sub]
models = set([s[0] for s in sub])
textures = set([s[1] for s in sub])

get_full_paths = lambda p: [osp.join(p, c) for c in os.listdir(p)]
outs = [o for o in get_full_paths('outputs') if '.' not in o]
outs = sum([get_full_paths(o) for o in outs], [])
outs = sum([get_full_paths(o) for o in outs], [])
keep = [o for o in outs if any([s in o for s in sub])]
nokeep = [o for o in outs if o not in keep]

# print(keep)
print(len(keep))

print(len(nokeep))

pending = [s for s in  sub if not any([s in k for k in keep])]
print('pending')
# print(pending)
print(len(pending))
for nk in nokeep:
    os.remove(nk)
