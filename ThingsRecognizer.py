import base64

import cv2
import numpy as np
import imutils
from starlette.responses import HTMLResponse

proto_path = "Files/MobileNetSSD_deploy.prototxt"
model_path = "Files/MobileNetSSD_deploy.caffemodel"
detection = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


def Recognize(type: str, img):
    obraz = cv2.imread(img)
    obraz = imutils.resize(obraz, width=600)

    (H, W) = obraz.shape[:2]

    blob = cv2.dnn.blobFromImage(obraz, 0.007843, (W, H), 127.5)

    detection.setInput(blob)
    people_counter = detection.forward()

    for i in np.arange(0, people_counter.shape[2]):
        accuracy = people_counter[0, 0, i, 2]
        if accuracy > 0.5:
            idx = int(people_counter[0, 0, i, 1])

            if CLASSES[idx] != type:
                continue

            square = people_counter[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = square.astype("int")

            cv2.rectangle(obraz, (startX, startY), (endX, endY), (0, 255, 0), 2)

    _, encoded = cv2.imencode('.PNG', obraz)
    encoded_img = base64.b64encode(encoded)

    html_content = """
    <div>
            <img src="data:image/png;base64, """ + encoded_img.decode("utf-8") + """" alt="Red dot" />
    </div>
    """

    return HTMLResponse(content=html_content, status_code=200)
