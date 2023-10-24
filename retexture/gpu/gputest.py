
import os
import os.path as osp
import sys

pkgs = [
        f"{osp.expanduser('~')}/.anaconda3/envs/retexture/lib/python3.11/site-packages",
        f"{osp.expanduser('~')}/miniconda3/envs/retexture/lib/python3.9/site-packages"
]
for pkg in pkgs:
    sys.path.insert(0, pkg)

import itertools
import math
from pprint import pprint

from tqdm import tqdm
import hydra

import bpy

ROOT = osp.dirname(osp.dirname(__file__))
CONFIGS = osp.join(ROOT, "configs")


def load(filepath):
    """TODO add a docstring"""

    if "Cube" in bpy.data.objects:  # delete default cube
        bpy.data.objects["Cube"].select_set(state=True)
        bpy.ops.object.delete()

    bpy.ops.wm.collada_import(filepath=filepath)


def ls_objs():
    """list all objects and determine if they are active"""

    def isactive(other):
        active = bpy.context.active_object
        return False if not active else active.name == other.name

    objs = {obj.name: isactive(obj) for obj in bpy.data.objects}
    print(objs)


def uvUnwrap(image_texture_path):
    """TODO docstring"""

    # Select and join all mesh objects
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    for obj in mesh_objects:  # cannot join 1st object to itself
        obj.select_set(state=True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.object.join()

    # Unwrap the object using smart UV unwrap
    obj = bpy.context.active_object
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode="OBJECT")

    # Convert Principled BSDF materials to Diffuse BSDF materials
    for slot, mat in enumerate(obj.material_slots):
        if mat.material.node_tree.nodes.get("Principled BSDF") is not None:
            principled_bsdf = mat.material.node_tree.nodes.get("Principled BSDF")
            diffuse_bsdf = mat.material.node_tree.nodes.new(type="ShaderNodeBsdfDiffuse")
            mat.material.node_tree.nodes.remove(principled_bsdf)
            mat.material.node_tree.links.new(
                diffuse_bsdf.outputs["BSDF"],
                mat.material.node_tree.nodes["Material Output"].inputs["Surface"],
            )

    # Set the texture for Diffuse BSDF materials
    for slot, mat in enumerate(obj.material_slots):
        if mat.material.node_tree.nodes.get("Diffuse BSDF") is not None:
            diffuse_bsdf = mat.material.node_tree.nodes.get("Diffuse BSDF")
            image_texture = mat.material.node_tree.nodes.new(type="ShaderNodeTexImage")
            image_texture.image = bpy.data.images.load(image_texture_path)
            mat.material.node_tree.links.new(
                image_texture.outputs["Color"], diffuse_bsdf.inputs["Color"]
            )

    # Switch to the UV Editing workspace
    uv_editing_workspace = bpy.data.workspaces.get("UV Editing")
    uv_editing_screen = uv_editing_workspace.screens.get("UV Editing")
    bpy.context.window.workspace = uv_editing_workspace

    # Select all elements in UV Editing
    if obj.type == "MESH":
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.select_mode(type="FACE")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.select_mode(type="EDGE")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.object.mode_set(mode="OBJECT")

    # Unwrap the UVs
    bpy.ops.object.mode_set(mode="EDIT")
    if obj.type == "MESH":
        bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode="OBJECT")


def calculate_angle(i, num_angles):
    return i * (2 * math.pi / num_angles)


def set_camera_location_and_lens(location, lens):
    """
    Adjust the Z-coordinate as needed #4x the max value of the shape(?),
    height maybe 1.1x just as dimensions to try starting with.
    Want prooption to be fixed by the size of the model. Length at the longest point ie xMax-xMin
    """
    bpy.context.scene.camera.location = location
    bpy.context.scene.camera.data.lens = lens


def center_camera_on_object():
    """
    outputs look like centering on center of mass, move camera way far back to see if centering gets better or worse.
    If better, center of mass issue is washed away
    """
    bpy.ops.view3d.camera_to_view_selected()


def set_object_rotation(rotation_euler):
    bpy.data.objects['SketchUp'].rotation_euler = rotation_euler


def render_image():
    bpy.ops.render.render(write_still=True)


def save_rendered_image(outpath, degrees, file_type):
    outpath = f"{outpath}_{degrees}.{file_type}"
    bpy.data.images["Render Result"].save_render(filepath=outpath)


def render_all(nangles, outpath, file_type):
    for i in range(nangles):
        radians = i * (2 * math.pi / nangles)
        degrees = i * (360 / nangles)

        set_camera_location_and_lens(location=(-32, -32, 10), lens=15)
        center_camera_on_object()
        set_object_rotation(rotation_euler=(0, 0, radians))
        render_image()
        save_rendered_image(outpath, degrees, file_type)


def load_data(cfg):
    """loads file paths for models and textures"""
    model_dir = osp.join(ROOT, cfg.data_dir, "models")
    tex_dir = osp.join(ROOT, cfg.data_dir, "textures")

    get_full_paths = lambda p: [osp.join(p, c) for c in os.listdir(p)]

    models = get_full_paths(model_dir)
    # it is a nested directory rn
    textures = [get_full_paths(x) for x in get_full_paths(tex_dir)]
    textures = sum(textures, [])
    return models, textures


def basename(file):
    """return filename with no extensions"""
    return osp.basename(file).split(".")[0]



def oldmain():
    """docstring"""

    hydra.initialize(version_base="1.2", config_path="../configs")
    cfg = hydra.compose(config_name="base")

    models, textures = load_data(cfg)
    pairs = itertools.product(models, textures)

    for i, (model, texture) in enumerate(pairs):
        print("i", i)

        outname = "_".join([basename(x) for x in (model, texture)]) 
        outpath = osp.join(cfg.out_dir, outname)

        load(model)
        uvUnwrap(texture)
        render_all(cfg.nangles, outpath,  cfg.file_type)


        bpy.data.objects["SketchUp"].select_set(state=True)
        bpy.ops.object.delete()



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
        models, textures = load_data(self.cfg)
        pairs = itertools.product(models, textures)
        return pairs

    def run(self):
        """runs all rendering steps"""

        for (model, texture) in tqdm(self.pairs):
            outname = "_".join([basename(x) for x in (model, texture)]) 
            outpath = osp.join(self.cfg.out_dir, outname)

            load(model)
            uvUnwrap(texture)
            render_all(self.cfg.nangles, outpath,  self.cfg.file_type)

            self.clear_scene()

            # BR.render_scene('/path/to/output.png')


def main():
    """docstring"""

    BR = ModelCapture()
    BR.configure_gpu_render()
    BR.run()


if __name__ == "__main__":
    main()
