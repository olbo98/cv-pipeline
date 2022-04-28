import numpy as np
import tensorflow as tf
from models import YoloV3
from mapcalc import calculate_map
def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train

def main():
    yolo = YoloV3(416, classes=80)
    yolo.load_weights("./checkpoints/yolov3.tf").expect_partial()
    labels = []
    boxess = []
    with open('./annotations/00000_FV.txt', 'r') as box:
        for line in box:
            label = line.split(" ")
            label = [float(x) for x in label]
            labels.append(label.pop())
            boxess.append(label)
    #print(type(boxess))
    gt = {'boxes': boxess, 'labels': labels}
    
    img_raw = tf.image.decode_image(
            open('./labeled_images/00000_FV.png', 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, 416)

    boxes, scores, classes, nums = yolo.predict(img)
    boxes = boxes.tolist()
    scores = scores.tolist()
    classes = classes.tolist()
    print(nums)
    print(labels)
    print(classes[0])
    print(scores)

    pred = {'boxes':boxes[0],'labels':classes[0], 'scores':scores[0]}
    print(calculate_map(gt, pred, 1))
    
if __name__ == "__main__":
    main()