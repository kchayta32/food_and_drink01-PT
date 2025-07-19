import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0)

model_path = 'my_images for food_& drink-seg.pt'
model = YOLO(model_path)

while cap.isOpened():
    ret, image = cap.read()
    image = cv2.flip(image, 1)

    key = cv2.waitKey(33)
    if key == ord('q') or ret == False:
        break

    results = model.predict(source=image, device='cpu', save=False)

    for result in results:
        output = result.plot()

        cv2.imshow('output prediction', output)

cap.release()
cv2.destroyAllWindows()
