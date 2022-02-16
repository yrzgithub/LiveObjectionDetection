from keyboard import is_pressed
from detecto import core
import cv2

model = core.Model()


def filter_top(img):  # object detection
    labels, boxes, scores = model.predict_top(img)
    scores = scores.tolist()
    boxes = boxes.tolist()  # convert tensor obj to list 2D
    new_labels = []
    new_boxes = []
    new_scores = []
    for i, j, k in zip(labels, boxes, scores):
        if k > 0.8:
            new_labels.append(i)
            new_boxes.append(j)
            new_scores.append(k)
    return new_labels, new_boxes


def start_live_camera():
    cam = cv2.VideoCapture(0)
    img = None
    while not is_pressed("esc"):
        ret, img = cam.read()
        if not ret:
            continue
        labels, new_boxes = filter_top(img)
        for i in new_boxes:
            (a, b, c, d) = map(int, i)
            cv2.rectangle(img, (a, b), (c, d), color=(250, 0, 250), thickness=2)  # doubt
        print(labels)
        cv2.imshow("object detection", img)
        cv2.waitKey(10)
    cv2.destroyAllWindows()
    cam.release()
    return img

start_live_camera()
