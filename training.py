import numpy as np, pandas as pd, matplotlib.pyplot as plt, cv2, pickle, json
from PIL import Image
import albumentations as A
from tqdm import tqdm
#start
import os, time, uuid, cv2
import tensorflow as tf

#uniform unique identifier(uuid)

IMAGES_PATH= os.path.join('aug_data', 'images')

def load_image(x):
    byte_img= tf.io.read_file(x)
    img= tf.io.decode_jpeg(byte_img)
    return img

train_images= tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False)
train_images= train_images.map(load_image)
train_images= train_images.map(lambda x: tf.image.resize(x, (120, 120)))
train_images= train_images.map(lambda x: x/255)

test_images= tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False)
test_images= test_images.map(load_image)
test_images= test_images.map(lambda x: tf.image.resize(x, (120, 120)))
test_images= test_images.map(lambda x: x/255)


valid_images= tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle=False)
valid_images= valid_images.map(load_image)
valid_images= valid_images.map(lambda x: tf.image.resize(x, (120, 120)))
valid_images= valid_images.map(lambda x: x/255)

def load_labels(label_path):
    
    with open(label_path.numpy(), 'r', encoding='utf-8') as f:
        label= json.load(f)
    return [label['class']], label['bbox']

train_labels= tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle=False)

train_labels= train_labels.map(lambda x:tf.py_function(load_labels, [x], [tf.uint8, tf.float16])) #py_func gets (function, func parameter, return type1, return type 2)

test_labels= tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
test_labels= test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

valid_labels= tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json', shuffle=False)
valid_labels= valid_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))


# print(len(train_images), len(train_labels), len(valid_images), len(valid_labels), len(test_images), len(test_labels))

#create Final_dataset

train= tf.data.Dataset.zip((train_images, train_labels))
train= train.shuffle(5000)
train= train.batch(8)
train= train.prefetch(4)

test= tf.data.Dataset.zip((test_images, test_labels))
test= test.shuffle(1300)
test= test.batch(8)
test= test.prefetch(4)

valid= tf.data.Dataset.zip((valid_images, valid_labels))
valid= valid.shuffle(1000)
valid= valid.batch(8)
valid= valid.prefetch(4)


data_samples= train.as_numpy_iterator()

print('data samples', data_samples)
res= data_samples.next()

X, y= res

print(X.shape, y)

from deeplearning import localization_loss, facetracker, FaceTracker, optmzr, classloss, regressionloss
# classes, coords= facetracker.predict(X)
# print(classes, '---', coords)

# print(localization_loss(y[1], coords).numpy())
model= FaceTracker(facetracker)

model.compile(optmzr, classloss, regressionloss)
logdir='logs'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    'model_checkpoint.h5', save_best_only=True)

tensorboard_callback= tf.keras.callbacks.TensorBoard(log_dir= logdir)
hist= model.fit(train, epochs=10, validation_data= valid, callbacks=[tensorboard_callback])
model.save_weights('mymodel_weights.tf', save_format='tf')