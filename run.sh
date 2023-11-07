blender -b -P $(dirname "$0")/retexture/main.py
python $(dirname "$0")/retexture/walk_media.py
