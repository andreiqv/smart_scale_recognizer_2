import tensorflow as tf
import cv2


model = tf.keras.models.load_model('/home/andrei/Data/Datasets/scales_temp/151018/se_recognizer131018.60-3.31.hdf5')



img = cv2.imread("./images/3.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(img, (299, 299), cv2.INTER_CUBIC)

img = img[..., :: -1]
img = img[None, ...]
img = img / 255.0

res = model.predict(img)
print(res)