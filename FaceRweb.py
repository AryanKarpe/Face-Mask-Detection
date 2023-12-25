import cv2
import numpy as np

vid = cv2.VideoCapture(0)

width, height = 320, 320
configThreshold = 0.5
classFile = 'coco.names'

className = []

with open(classFile, 'rt') as f:
    className = f.read().rstrip('\n').split('\n')

modelConfig = 'custom-yolov4-tiny-detector.cfg'
modelWeight = 'custom-yolov4-tiny-detector_best.weights'

net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObject(outputs, img):
    hgt, wdt, cT = img.shape
    boundingBox = []
    classID = []
    configs = []

    for output in outputs:
        for detect in output:
            score = detect[5:]
            classId = np.argmax(score)
            confidence = score[classId]
            if confidence > configThreshold:
                w, h = int(detect[2] * wdt), int(detect[3] * hgt)
                x, y = int((detect[0] * wdt) - w / 2), int((detect[1] * hgt) - h / 2)
                boundingBox.append([x, y, w, h])
                classID.append(classId)
                configs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boundingBox, configs, configThreshold, nms_threshold=0.3)

    for i in indices:
        i = i
        box = boundingBox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{className[classID[i]].upper()} {int(configs[i] * 100)}%', (x, y - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

while True:
    success, img = vid.read()

    blob = cv2.dnn.blobFromImage(img, 1/255, (width, height), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()

    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)

    findObject(outputs, img)
    cv2.imshow('Face mask detection', img)

    key = cv2.waitKey(1)  

    if key == 27:
        break

vid.release()
cv2.destroyAllWindows()
