import media
import os
import os.path as osp
import argparse


def find_terminal_folders(root_path):
    for dirpath, dirnames, files in os.walk(root_path):
        # If dirnames is empty, this is a terminal folder
        if not dirnames:
            yield dirpath


def main():

    dn = osp.dirname
    output = osp.join(dn(dn(dn(__file__))),'outputs')
    assert osp.exists(output) 

    pairs = list(find_terminal_folders(output))
    for p in pairs:
        paths = [x for  x in list(os.walk(p))[0][2] if '.png' in x]
        paths = [osp.join(p,x) for x in paths]
        media.compose_gif(paths, outdir=p)
        media.compose_video(paths, outdir=p)

if __name__ == "__main__":
    main()
