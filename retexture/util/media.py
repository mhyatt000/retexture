import argparse
import os
import os.path as osp
import imageio
import re

import moviepy.editor as mp


def frame2arr(paths):
    """ reads an array of paths to np frames """
    frames = [imageio.imread(path) for path in paths]
    return frames

def sort_paths(paths):
    return sorted(paths, key=lambda x: float(re.search(r'(\d+\.?\d*)\.png$', x).group(1)))

def compose_gif(paths, outdir=None, fps=1):
    frames = frame2arr(paths)
    out = osp.join(outdir,'output.gif') if outdir else 'output.gif'
    imageio.mimsave(out, frames, 'GIF', duration=1/fps)


def compose_video(paths, fps=4, outdir=None):
    clips = [mp.ImageClip(frame).set_duration(1/fps) for frame in paths]
    final_clip = mp.concatenate_videoclips(clips, method="compose")
    final_clip.fps = fps

    out = osp.join(outdir,'output.mp4') if outdir else 'output.mp4'
    final_clip.write_videofile(out, fps=fps)

def main():
    parser = argparse.ArgumentParser(description="Compose a video from a list of video paths.")
    parser.add_argument("paths", nargs="+", help="Paths to video files")
    parser.add_argument("--fps", type=float, default=1,  help="Video fps")
    parser.add_argument("--nosort", action="store_true",   help="sort paths by num? default=sort")
    args = parser.parse_args()

    paths = sort_paths(args.paths) if not args.nosort else args.paths
    compose_video(paths, fps=args.fps)
    compose_gif(args.paths, fps=args.fps)

if __name__ == "__main__":
    main()
