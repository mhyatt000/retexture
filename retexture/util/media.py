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

def compose_gif(paths, outdir=None):
    paths = sorted(paths, key=lambda x: float(re.search(r'(\d+\.?\d*)\.png$', x).group(1)))
    frames = frame2arr(paths)
    out = osp.join(outdir,'output.gif') if outdir else 'output.gif'
    imageio.mimsave(out, frames, 'GIF', duration=0.1)


def compose_video(paths, framerate=4, outdir=None):
    paths = sorted(paths, key=lambda x: float(re.search(r'(\d+\.?\d*)\.png$', x).group(1)))
    clips = [mp.ImageClip(frame).set_duration(1/framerate) for frame in paths]
    final_clip = mp.concatenate_videoclips(clips, method="compose")
    final_clip.fps = framerate

    out = osp.join(outdir,'output.mp4') if outdir else 'output.mp4'
    final_clip.write_videofile(out, fps=framerate)

def main():
    parser = argparse.ArgumentParser(description="Compose a video from a list of video paths.")
    parser.add_argument("paths", nargs="+", help="Paths to video files")
    parser.add_argument("--framerate", type=int, default=4,  help="Video framerate")
    args = parser.parse_args()

    compose_video(args.paths, framerate=args.framerate)
    compose_gif(args.paths)

if __name__ == "__main__":
    main()
