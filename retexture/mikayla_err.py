""" helllloooo my friends here is the issue I am running into.... 
so i have the code to resize the models to be close to the size of elephant2, which is the model that the camera coordinates are based on. that code is as follows: 
"""

import bpy
for o in bpy.data.objects:
    print(o)
obj = bpy.data.objects['SketchUp']
new_scale = (22, 22, 22)
obj.scale = new_scale
new_dimensions = (new_scale[0] * obj.dimensions.x, new_scale[1] * obj.dimensions.y, new_scale[2] * obj.dimensions.z)
obj.dimensions = new_dimensions

""" so now i am trying to move the active object to the origin such that the camera can capture the model from every viewpoint, but the following code is not working:  """

object_name = "YourObjectName"
target_x = 2.0  # Replace with your desired X coordinate
target_y = 3.0  # Replace with your desired Y coordinate
target_z = 1.0  # Replace with your desired Z coordinate
# Find the object by name
obj = bpy.data.objects.get(object_name)
if obj:
    # Set the object's location to the target coordinates
    obj.location = (target_x, target_y, target_z)

"""
any suggestions as to how to why this is not working? nothing changes when i run the code to center the object about the origin, but the code to change the size of the model is working 
also also here is the code for the camera:
"""

# Loop through each angle and render the image
for i in range(num_angles):
    # Calculate the angle to rotate the object
    angle = i * (2*math.pi/num_angles)
    bpy.context.scene.camera.location = (25, -25, 25)
    # Change the camera's focal length (in millimeters)
    bpy.context.scene.camera.data.lens = 20
    # Select the object to rotate
    obj = bpy.data.objects[object_name]
    # Set the rotation of the object
    obj.rotation_euler = (0, 0, angle)
    # Render the image
    bpy.ops.render.render(write_still=True)
    angleName = i * (360/num_angles)
    # Set the file name for the rendered image
    file_name = "{}{}{}".format(base_file_name, angleName, file_type)
    # Save the rendered image to the output directory
    bpy.data.images['Render Result'].save_render(filepath=os.path.join(output_directory, file_name))
