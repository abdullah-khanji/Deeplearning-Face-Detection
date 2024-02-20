import os, matplotlib.pyplot as plt
from PIL import Image

IMAGES_PATH= os.path.join('data', 'images')

images_names= os.listdir(IMAGES_PATH)

images=[Image.open(os.path.join(IMAGES_PATH, i)) for i in images_names]
print(images[1:4])
# Display images
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))  # Adjust the size as needed
axes = axes.flatten()

for img, ax in zip(images, axes):
    ax.imshow(img)
    ax.axis('off')  # Hide axes

plt.tight_layout()
plt.show()

#for tensor of images
# plot_images= image_gen.next()

# # Display images
# fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(20, 8))  # Adjust the size as needed
# # fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))  # Adjust the size as needed
# ax = ax.flatten()
# for idx, image in enumerate(plot_images):
#     ax[idx].imshow(image)


# plt.tight_layout()
# plt.show()
