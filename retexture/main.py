""" Utils for blender """

import argparse
import itertools
import json
import math
import os
import os.path as osp
import sys
from contextlib import contextmanager
from pprint import pprint

import bmesh
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

ROOT = osp.dirname(osp.dirname(__file__))
CONFIGS = osp.join(ROOT, "configs")


def get_args():
    """Get arguments after '--'"""

    # Check if '--' is in the arguments, otherwise just parse the command line
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1 :]
    else:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Directory for data")
    parser.add_argument("--out_dir", help="Output directory")
    parser.add_argument("--nangles", type=int, help="Number of angles")
    parser.add_argument("--file_type", help="Type of the rendered images")
    parser.add_argument("--silent", action="store_true", help="Run in silent mode")

    parser.add_argument("--model", help="Directory for data")
    parser.add_argument("--texture", help="Directory for data")
    return parser.parse_args(args)


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

    objs = {obj.name: (obj.type, isactive(obj)) for obj in bpy.data.objects}
    print(objs)


def uvUnwrap(image_texture_path, outname):
    """TODO docstring"""

    # Select and join all mesh objects
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    for obj in mesh_objects:  # cannot join 1st object to itself
        obj.select_set(state=True)

    # Set the active object to the first in the list and rename it
    # sometimes the mesh is not called sketch up if the model name has
    # a lesser alphabetic name
    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.context.view_layer.objects.active.name = outname
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
            diffuse_bsdf = mat.material.node_tree.nodes.new(
                type="ShaderNodeBsdfDiffuse"
            )
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


def center_mesh(outname):
    """
    gets a translation vector by finding the middle between min/max in xyz
    then subtracts this vector to center the object
    """

    obj = bpy.data.objects[outname]

    # remove its parents if there are any
    obj.parent = None

    if obj.type != "MESH":
        raise TypeError(f"The object must be of type 'MESH' but is {obj.type}")

    mesh = obj.data
    bpy.context.view_layer.update()

    # Calculate the min and max coordinates for each axis
    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")

    # !!!
    # you have to do this iteratively since bpy uses custom data structures
    for vert in mesh.vertices:
        min_x, min_y, min_z = (
            min(min_x, vert.co.x),
            min(min_y, vert.co.y),
            min(min_z, vert.co.z),
        )
        max_x, max_y, max_z = (
            max(max_x, vert.co.x),
            max(max_y, vert.co.y),
            max(max_z, vert.co.z),
        )

    # Calculate the midpoints
    mid_x = (max_x + min_x) / 2
    mid_y = (max_y + min_y) / 2
    mid_z = (max_z + min_z) / 2

    # Translate vertices so that the midpoints are at the origin
    translation_vector = Vector((-mid_x, -mid_y, -mid_z))
    for vert in mesh.vertices:
        vert.co += translation_vector


def convert_to_mesh(object_name):
    # Retrieve the object
    obj = bpy.data.objects[object_name]

    # Make sure we're in object mode
    bpy.ops.object.mode_set(mode="OBJECT")

    # Select the object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Convert the object to a mesh
    bpy.ops.object.convert(target="MESH")


def calculate_angle(i, num_angles):
    return i * (2 * math.pi / num_angles)


def delete_other_cameras():
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.data.objects:
        if obj.type == "CAMERA" and obj.name != "Camera":
            obj.select_set(True)
    bpy.ops.object.delete()


def set_camera_location_and_lens(location, lens):
    """
    Adjust the Z-coordinate as needed #4x the max value of the shape(?),
    height maybe 1.1x just as dimensions to try starting with.
    Want prooption to be fixed by the size of the model. Length at the shortest point ie xMax-xMin
    """
    delete_other_cameras()
    bpy.context.scene.camera.location = location
    bpy.context.scene.camera.data.lens = lens


def set_object_rotation(outname, rotation_euler):
    bpy.data.objects[outname].rotation_euler = rotation_euler


def render_image():
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (
        1,
        1,
        1,
        1,
    )  # RGBA
    bpy.ops.render.render(write_still=True)


def save_rendered_image(outpath, degrees, file_type):
    outpath = f"{outpath}_{degrees}.{file_type}"
    bpy.data.images["Render Result"].save_render(filepath=outpath)


def apply_modifiers(obj):
    """sometimes you need to be in object mode to transform an object"""

    # Make sure we're in Object mode
    bpy.ops.object.mode_set(mode="OBJECT")

    # Apply all modifiers for the object
    while obj.modifiers:
        bpy.ops.object.modifier_apply(modifier=obj.modifiers[0].name)


def set_origin2mass(obj):
    # Make sure we're in Object mode
    bpy.ops.object.mode_set(mode="OBJECT")

    # Select the object
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Set the origin to the center of mass
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS")


def point2origin():
    camera = bpy.data.objects["Camera"]
    # Calculate the direction from the camera to the target
    direction = Vector((0, 0, 0)) - camera.location
    # Point the camera's '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat("-Z", "Y")
    # Convert the quaternion to euler angles
    camera.rotation_euler = rot_quat.to_euler()


def is_in_view(camera_obj, mesh_obj):
    """checks if mesh is within camera frame entirely"""

    scene = bpy.context.scene
    cam_data = camera_obj.data

    # Get the transformation matrix from the world to the camera
    mat_world_to_camera = camera_obj.matrix_world.inverted()

    # Check if mesh vertices are in the camera view
    for vertex in mesh_obj.data.vertices:
        # Transform the vertex to world space
        world_vertex = mesh_obj.matrix_world @ vertex.co
        # Use the utility function world_to_camera_view
        camera_vertex = world_to_camera_view(scene, camera_obj, world_vertex)
        # Check if the camera vertex is within the view frustum (0.0 to 1.0 means it is visible)
        if not (
            0.0 <= camera_vertex.x <= 1.0
            and 0.0 <= camera_vertex.y <= 1.0
            and camera_vertex.z > 0
        ):
            return False

    return True


def fit2view(obj):
    """fits an object to the maximum camera view"""

    obj.scale = (1.0, 1.0, 1.0)
    toosmall = is_in_view(bpy.data.objects["Camera"], obj)

    if toosmall:
        while toosmall:
            obj.scale *= 1.1
            toosmall = is_in_view(bpy.data.objects["Camera"], obj)
            print(obj.scale)
    else:  # too large
        while not toosmall:
            obj.scale *= 0.9
            toosmall = is_in_view(bpy.data.objects["Camera"], obj)
            print(obj.scale)


# Create a visual marker for the origin
def create_origin_marker(size=1, display_type="SOLID"):
    """HELPER
    makes a sphere at origin of something
    """
    bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=(0, 0, 0))
    origin_marker = bpy.context.active_object
    origin_marker.display_type = display_type
    origin_marker.name = "Origin Marker"
    return origin_marker


# Parent the marker to an object to represent its local origin
def parent_marker_to_object(marker, obj):
    """HELPER
    moves a temporary marker to become the child of obj
    """
    marker.parent = obj
    marker.matrix_parent_inverse = obj.matrix_world.inverted()


def normalize_object_scales(obj):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")

    dims = obj.dimensions
    ref_dims = [15, 15, 20]  # maximum size in x,y,z
    scales = [r / d for r, d in zip(ref_dims, dims)]
    print(scales)
    obj.scale *= min(scales)  # scale everything equally by minimum allowed


def render_all(nangles, outname, outpath, file_type):
    obj = bpy.data.objects[outname]
    normalize_object_scales(obj)
    for i in range(nangles):
        radians = i * (2 * math.pi / nangles)
        degrees = i * (360 / nangles)

        # apply_modifiers( bpy.data.objects[outname])
        set_object_rotation(outname, rotation_euler=(0, 0, radians))
        # center_mesh(outname)

        set_origin2mass(obj)
        obj.location = (0, 0, 0)

        # fit2view(obj)
        set_camera_location_and_lens(location=(10, 10, 10), lens=15)
        point2origin()

        render_image()
        save_rendered_image(outpath, degrees, file_type)


def load_data(args):
    """loads file paths for models and textures"""
    model_dir = osp.join(ROOT, args.data_dir, "models")
    tex_dir = osp.join(ROOT, args.data_dir, "textures")

    get_full_paths = lambda p: [osp.join(p, c) for c in os.listdir(p)]

    models = get_full_paths(model_dir)
    textures = get_full_paths(tex_dir)
    return models, textures


def basename(file):
    """return filename with no extensions"""
    return osp.basename(file).split(".")[0]


def main():
    """docstring"""

    args = get_args()

    outname = "_".join([basename(x) for x in (args.model, args.texture)])
    parent = osp.join(args.out_dir, basename(args.model), basename(args.texture))
    outpath = osp.join(parent, outname)

    if not osp.exists(parent) or len(os.listdir(parent)) != args.nangles:
        load(args.model)
        uvUnwrap(args.texture, outname)
        render_all(args.nangles, outname, outpath, args.file_type)

        bpy.data.objects[outname].select_set(state=True)
        bpy.ops.object.delete()

    with open(osp.join(args.out_dir, "err.json"), "w") as file:
        json.dump(err, file)


if __name__ == "__main__":
    main()
