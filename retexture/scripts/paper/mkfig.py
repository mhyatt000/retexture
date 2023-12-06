import os

import matplotlib.pyplot as plt
from PIL import Image

# List all image files in the current directory
image_files = [
    file
    for file in os.listdir(".")
    if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
]

# Sort the files to ensure consistent order
# image_files.sort()

# Plotting the images in a 2x8 grid
fig, axes = plt.subplots(2, 8, figsize=(40, 10))
axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing

for i, file in enumerate(image_files[:16]):
    # Load the image
    img = Image.open(file)

    # Crop the image so that the width matches the height
    width, height = img.size
    left = (width - height) / 2
    top = 0
    right = (width + height) / 2
    bottom = height
    img_cropped = img.crop((left, top, right, bottom))

    # Plot the cropped image
    ax = axes[i]
    ax.imshow(img_cropped)
    ax.axis("off")  # Turn off axis


# Remove spaces between axes
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
# plt.show()
plt.savefig("display.png")
