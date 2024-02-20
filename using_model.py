import os
import tensorflow as tf, json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1=WARN, 2=ERROR, 3=FATAL)

tf.get_logger().setLevel('ERROR')  # 
import tensorflow as tf, numpy as np, cv2
from deeplearning import build_model, FaceTracker

model = FaceTracker(build_model())  # Rebuild the model architecture
model.load_weights('mymodel_weights.tf')  # Load the saved weights

def load_image(x):
    byte_img= tf.io.read_file(x)
    img= tf.io.decode_jpeg(byte_img)
    return img


def load_labels(label_path):
    
    with open(label_path.numpy(), 'r', encoding='utf-8') as f:
        label= json.load(f)
    return [label['class']], label['bbox']


#===> this is for testing model with test data--------------<----------------

# test_images= tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False)
# test_images= test_images.map(load_image)
# test_images= test_images.map(lambda x: tf.image.resize(x, (120, 120)))
# test_images= test_images.map(lambda x: x/255)

# test_labels= tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
# test_labels= test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

# test= tf.data.Dataset.zip((test_images, test_labels))
# test= test.shuffle(1300)
# test= test.batch(8)
# test= test.prefetch(4)

# test_data= test.as_numpy_iterator()
# test_sample= test_data.next()

# yhat= model.predict(test_sample[0])

# fig, ax= plt.subplots(ncols=4, figsize=(20, 20))
# for idx in range(4):
#     sample_image= test_sample[0][idx]
#     sample_coords= yhat[1][idx]

#     if yhat[0][idx]>0.5:
#         cv2.rectangle(sample_image, 
#                       tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
#                       tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
#                       (255,20, 4), 2
#                       )
    
#     ax[idx].imshow(sample_image)

# plt.show()

cap= cv2.VideoCapture(0)

while cap.isOpened():
    _, frame= cap.read()
    frame= frame[50:500, 50:500, :]

    rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized= tf.image.resize(rgb, (120, 120))
    
    yhat= model.predict(np.expand_dims(resized/255, 0))

    sample_coords=yhat[1][0]

    if yhat[0]>0.5:
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                      (255, 0, 0), 2)
        
        
        cv2.putText(frame, 'face', 
                    tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [0, -5])),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                    )
    cv2.imshow('Face Track', frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


print("just checking")

print("just checking")

print("just checking")