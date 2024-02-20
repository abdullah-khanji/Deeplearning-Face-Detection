# for data
import time, uuid, os, cv2

number_images=30
IMAGES_PATH= os.path.join('data', 'images')


cap= cv2.VideoCapture(0)

for imgnum in range(number_images):
  print('Collecting image {}'.format(imgnum))
  ret, frame= cap.read()
  if not ret:
    print("sorry")
    break
  imgname= os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
  print(imgname,'-----')
  cv2.imwrite(imgname, frame)
  cv2.imshow('frame', frame)
  time.sleep(0.8)

  if cv2.waitKey(1) & 0xFF==ord('q'):
    break

cap.release()
cv2.destroyAllWindows()