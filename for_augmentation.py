import albumentations as A, os, cv2, matplotlib.pyplot as plt, json, tqdm, numpy as np

augmentor = A.Compose([A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomGamma(p=0.2),
    A.RGBShift(p=0.2),
    A.VerticalFlip(p=0.5),
], bbox_params=A.BboxParams(format='albumentations',
                            label_fields=['class_labels']))

def albument(img1, partition):
    img= os.path.join('data', partition, 'images',img1)
    label_path= os.path.join('data', partition, 'labels', (img1.split('.')[0])+'.json')
    image= cv2.imread(img)
    coords=[0, 0, 0, 0]
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            label= json.load(f)
    
        coords[0]= label['shapes'][0]['points'][0][0]
        coords[1]= label['shapes'][0]['points'][0][1]
        coords[2]= label['shapes'][0]['points'][1][0]
        coords[3]= label['shapes'][0]['points'][1][1]
        coords= list(np.divide(coords, [640, 480, 640, 480]))

    try:
        for x in tqdm(range(60)):
            augmented= augmentor(image= image, bboxes=[coords], class_labels=['face'])
            cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{img1.split(".")[0]}.{x}.jpg'), augmented['image'])
            annotation= {}
            annotation['image']=img1

            if os.path.exists(label_path):
                if len(augmented['bboxes'])==0:
                    annotation['bbox']= [0, 0, 0, 0]
                    annotation['class']= 0
                
                else:
                    annotation['bbox']= augmented['bboxes'][0]
                    annotation['class']=1
            else:
                annotation['bbox']=[0, 0, 0, 0]
                annotation['class']=0
            with open(os.path.join('aug_data', partition, 'labels', f'{img1.split(".")[0]}.{x}.json'), 'w+') as f:
                json.dump(annotation, f)
                
    except Exception as e:
        print(e)

for partition in ['train','test', 'val']:
    for image in os.listdir(os.path.join('data', partition, 'images')):
        albument(image, partition)