import cv2, json, os, numpy as np, matplotlib.pyplot as plt, random

base_dir= os.path.join('aug_data', 'train', 'images')

images_names=os.listdir(base_dir)

random.shuffle(images_names)

print(images_names[0:11])

labels=[]
images=[]
for name in images_names[0:12]:
    path= os.path.join(base_dir, name)
    label_path= os.path.join('aug_data', 'train', 'labels', ('.'.join(name.split('.')[0:2])+'.json'))
    images.append(cv2.imread(path))

    with open(label_path, 'r') as f:
        labels.append(json.load(f))
    
    
import matplotlib.patches as patches

fig, axs = plt.subplots(4, 3, figsize=(15, 20))  # 4 rows, 3 columns to display 12 images
axs = axs.ravel()  # Flatten the array for easy indexing



for i in range(0, 12):
    label=labels[i]
    img=images[i]
    axs[i].imshow(img)

    bbox= np.multiply(label['bbox'], [450, 450, 450, 450]).astype(int)
    top_left= tuple(bbox[:2])
    width= bbox[2]-bbox[0]
    height= bbox[3]- bbox[1]
    # Create a rectangle patch
    rect = patches.Rectangle(top_left, width, height, edgecolor=(1, 1, 0), facecolor='none', linewidth=2)
    axs[i].add_patch(rect)  # Add the rectangle to the plot

    axs[i].axis('off')  # Turn off axis for cleaner look

plt.tight_layout()
plt.show()

