import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import time

model_path = 'latest_model_float32.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(output_details)

# Obtain the height and width of the corresponding image from the input tensor
image_height = input_details[0]['shape'][1] # 640
image_width = input_details[0]['shape'][2] # 640

# Image Preparation
image_name = 'test1.jpeg'
image = Image.open(image_name)
image_resized = image.resize((image_width, image_height)) # Resize the image to the corresponding size of the input tensor and store it in a new variable

image_np = np.array(image_resized) #
image_np = np.true_divide(image_np, 255, dtype=np.float32) 
image_np = image_np[np.newaxis, :]

# inference
interpreter.set_tensor(input_details[0]['index'], image_np)

start = time.time()
interpreter.invoke()
print(f'run time：{time.time() - start:.2f}s')

# Obtaining output results
output = interpreter.get_tensor(output_details[0]['index'])
output = output[0]
print(output.shape)
# Threshold Setting
threshold = 0.9

# Bounding boxes, scores, and classes are drawn on the image
draw = ImageDraw.Draw(image_resized)
max_score = 0
max_detection = None

for i in range(output.shape[1]):
    detection = output[:, i]
    score = np.max(detection[4:])
    if score > max_score:
        max_score = score
        max_detection = detection

# Drawing the bounding box with the maximum score
if max_detection is not None and max_score >= threshold:
    x_center, y_center, width, height = max_detection[:4]
    x1 = int(max(0, (x_center - width / 2) * image_width))
    y1 = int(max(0, (y_center - height / 2) * image_height))
    x2 = int(min(image_width - 1, (x_center + width / 2) * image_width))
    y2 = int(min(image_height - 1, (y_center + height / 2) * image_height))

    cls = np.argmax(max_detection[4:])
    text = f"Class: {cls}, Score: {max_score:.2f}"
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    draw.text((x1, y1), text, fill="red")

# for i in range(output.shape[1]):
#     detection = output[:, i]
#     x_center, y_center, width, height = detection[:4]

#     # Convert to top-left corner coordinates and scale
#     x1 = int(max(0, (x_center - width / 2) * image_width))
#     y1 = int(max(0, (y_center - height / 2) * image_height))
#     x2 = int(min(image_width - 1, (x_center + width / 2) * image_width))
#     y2 = int(min(image_height - 1, (y_center + height / 2) * image_height))

#     # Get class and score
#     score = np.max(detection[4:])
#     cls = np.argmax(detection[4:])

#     if score >= threshold:
#         draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
#         text = f"Class: {cls}, Score: {score:.2f}"
#         draw.text((x1, y1), text, fill="red")

# Saving Images
image_resized.save(f"output/detected_{image_name}")