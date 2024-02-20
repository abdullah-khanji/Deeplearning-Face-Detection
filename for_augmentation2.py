import albumentations as A, os, cv2, tqdm, matplotlib.pyplot as plt, json, numpy as np


augmentor= A.compose([
    A.RandomCrop(width=450, height=450),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomGamma(p=0.2), 
    A.RGBShift(p=0.2),
    A.VerticalFlip(p=0.5)
], bbox_params= A.BboxParams(format='albumentations', label_fields=['class_labels']))

def albument(img):
    label_path=img.split('.')[0]+'.json'
    image= cv2.imread(img)
    coords=[0, 0, 0, 0]

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            label=json.load(f)

        coords[0]= label['shapes'][0]['points'][0][0]
        coords[1]= label['shapes'][0]['points'][0][1]
        coords[2]= label['shapes'][0]['points'][1][0]
        coords[3]= label['shapes'][0]['points'][1][1]

        coords=list(np.divide(coords, [640, 480, 640, 480]))
    try:
        for x in tqdm(3):
            augmented= augmentor(image= image, bboxes=[coords], class_labels=['face'])
            cv2.imwrite(os.path.join('aug_data2', f"{img.split('.')[0]}.{x}.jpg"))
            annotation= {}
            annotation['image']= img
            if os.path.exists(label_path):
                if len(augmented['bboxes'])==0:
                    annotation['bbox']