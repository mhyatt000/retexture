rm -r outputs
blender -b -P ~/cs/retexture/retexture/main.py
python ~/cs/retexture/retexture/util/walk_media.py
