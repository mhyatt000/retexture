import bpy

class BlenderGPURender:
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

if __name__ == "__main__":
    renderer = BlenderGPURender()
    renderer.setup_scene()
    renderer.configure_gpu_render()
    renderer.render_scene('output.png')

