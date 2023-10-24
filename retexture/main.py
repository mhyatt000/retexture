import argparse
import subprocess
import os
import os
import os.path as osp

# import hydra
# from omegaconf import DictConfig

ROOT = osp.dirname(osp.dirname(__file__))
CONFIGS = osp.join(ROOT, "configs")

def main():
    """runs util.py"""

    # Call the child script
    subprocess.run(['python', osp.join(osp.dirname(__file__),'util.py')])

# @hydra.main(version_base='1.2', config_path=CONFIGS, config_name='base') 
# def main(cfg):
    # pass

if __name__ == "__main__":
    main()
