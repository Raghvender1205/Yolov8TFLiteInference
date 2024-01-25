import tensorflow_hub as hub
import tensorflow as tf
import cv2

image = cv2.imread("test3.png")
image_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
image_tensor = tf.expand_dims(image_tensor, axis=0)
print(image_tensor)
# Apply image detector on a single image.
detector = hub.load("https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/frameworks/TensorFlow2/variations/fpnlite-320x320/versions/1")
detector_output = detector(image_tensor)
class_ids = detector_output["detection_classes"]

print(detector_output)