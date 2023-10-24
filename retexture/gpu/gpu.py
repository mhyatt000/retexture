
import os
import os.path as osp
import sys

pkgs = [
        f"{osp.expanduser('~')}/.anaconda3/envs/retexture/lib/python3.11/site-packages",
        f"{osp.expanduser('~')}/miniconda3/envs/retexture/lib/python3.9/site-packages"
]
for pkg in pkgs:
    sys.path.insert(0, pkg)

import retexture.main as master
import bpy
from tqdm import tqdm

class BlenderGPURender():
    def __init__(self):
        self.clear_scene()

    def clear_scene(self):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.delete()

    def setup_scene(self):
        bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, location=(0, 0, 0))

    def configure_gpu_render(self):
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        
        prefs = bpy.context.preferences
        cprefs = prefs.addons['cycles'].preferences
        
        # Enable CUDA
        cprefs.compute_device_type = 'CUDA'
        
        for device in cprefs.devices:
            if device.type == 'CUDA':
                device.use = True
                
    def render_scene(self, output_path):
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)

class ModelCapture(BlenderGPURender):
    """handles retexturing the models"""

    def __init__(self):
        super().__init__()

        hydra.initialize(version_base="1.2", config_path="../configs")
        self.cfg = hydra.compose(config_name="base")
        self.pairs = self.load_data()

    def load_data(self):
        """docstring"""
        models, textures = master.load_data(self.cfg)
        pairs = itertools.product(models, textures)
        return pairs

    def run(self):
        """runs all rendering steps"""

        for (model, texture) in tqdm(self.pairs):
            outname = "_".join([basename(x) for x in (model, texture)]) 
            outpath = osp.join(self.cfg.out_dir, outname)

            master.load(model)
            master.uvUnwrap(texture)
            master.render_all(self.cfg.nangles, outpath,  self.cfg.file_type)

            self.clear_scene()

            BR.render_scene('/path/to/output.png')


def main():
    """docstring"""

    BR = BlenderGPURender()
    BR.configure_gpu_render()
    BR.run()


if __name__ == "__main__":
    main()
