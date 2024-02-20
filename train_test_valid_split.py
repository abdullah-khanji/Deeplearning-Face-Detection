print(98*.7, 'train')
print(98*.15, 'test')
import os, random

files= os.listdir(os.path.join('data','images'))
images_paths=[os.path.join('data','images', im) for im in files]

random.shuffle(images_paths)

train= images_paths[0:67] 
# change the number according to your data eg: if you have 500 images then 500*70/100= 350 will be for training 
#remaining will be for val and test.

images_paths= images_paths[67:]
random.shuffle(images_paths)

med_= len(images_paths)//2
test= images_paths[:med_]
val= images_paths[med_:]

import shutil
from itertools import zip_longest

for tr, ts, vl in zip_longest(train, test, val):
    if tr is not None:
        shutil.move(tr, os.path.join('data','train','images'))
    if ts is not None:
        # test1= os.path.join('data','images', ts)
        shutil.move(ts, os.path.join('data','test','images'))
    if vl is not None:
        # valid1= os.path.join('data','images', vl)
        shutil.move(vl, os.path.join('data','val','images'))

for folder in ['train', 'test', 'val']:
    for file in os.listdir(os.path.join('data', folder, 'images')):
        filename= file.split('.')[0]+'.json'
        existing_filepath= os.path.join('data', 'labels', filename)
        if os.path.exists(existing_filepath):
            new_filepath= os.path.join('data', folder, 'labels', filename)
            os.replace(existing_filepath, new_filepath)