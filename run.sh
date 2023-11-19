rm -r outputs
blender -b -P ~/cs/retexture/retexture/main.py > /dev/null 2>&1
python ~/cs/retexture/retexture/util/walk_media.py
