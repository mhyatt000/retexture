import argparse
import re

import moviepy.editor as mp



def compose_video(frame_paths, framerate=4):
    frame_paths = sorted(frame_paths, key=lambda x: int(re.search(r'(\d+)\.png$', x).group(1)))
    clips = [mp.ImageClip(frame).set_duration(1/framerate) for frame in frame_paths]
    final_clip = mp.concatenate_videoclips(clips, method="compose")
    final_clip.fps = framerate
    final_clip.write_videofile("output.mp4", fps=framerate)

def main():
    parser = argparse.ArgumentParser(description="Compose a video from a list of video paths.")
    parser.add_argument("paths", nargs="+", help="Paths to video files")
    parser.add_argument("--framerate", type=int, default=4,  help="Video framerate")
    args = parser.parse_args()

    compose_video(args.paths, framerate=args.framerate)


if __name__ == "__main__":
    main()
