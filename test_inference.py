import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

model_path = 'latest_model_float32.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image_name = "test3.png"
image = Image.open(image_name)
image = image.resize((512, 512))
image = np.array(image, dtype=np.float32) / 255.
image = image[np.newaxis, ...]

interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
boxes = output[..., :4] # bbox
conf = output[..., 4:5] # conf
class_prob = output[..., 5:] # class prob

detections = []
for i in range(len(output)):
    for j in range(output.shape[1]):
        # obtain bbox
        x_min, y_min, x_max, y_max = boxes[i][j]
        # obtain class prob
        class_ = class_prob[i][j]
        class_id = np.argmax(class_prob)
        detections.append([conf[i][j], class_id, x_min, y_min, x_max, y_max])

print(detections)