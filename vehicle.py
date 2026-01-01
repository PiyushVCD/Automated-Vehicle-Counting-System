# vehicle Project

import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

# filtering parameters
min_width_rectangle = 100
min_height_rectangle = 100
min_contour_area = 2500

countlineposition = 550
offset = 6

algo = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40)

def center_handle(x, y, w, h):
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return cx, cy

detect = []
counter = 0

while True:
    ret, frame1 = cap.read()
    if not ret:
        break

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    img_sub = algo.apply(blur)
    _, img_sub = cv2.threshold(img_sub, 200, 255, cv2.THRESH_BINARY)

    dilat = cv2.dilate(img_sub, np.ones((5, 5), np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        dilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.line(
        frame1,
        (25, countlineposition),
        (1200, countlineposition),
        (255, 127, 0),
        3
    )

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_contour_area:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        if w < min_width_rectangle or h < min_height_rectangle:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

    # 🔑 SAFE iteration over copy of list
    for (cx, cy) in detect[:]:
        if (countlineposition - offset) < cy < (countlineposition + offset):
            counter += 1
            detect.remove((cx, cy))

            # ✅ TERMINAL OUTPUT (NOW GUARANTEED)
            print(f"Vehicle {counter} passed the line")

            # video annotation
            cv2.putText(
                frame1,
                f"Vehicle {counter}",
                (cx - 20, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.line(
                frame1,
                (25, countlineposition),
                (1200, countlineposition),
                (0, 127, 255),
                3
            )

    cv2.putText(
        frame1,
        "Vehicle Count: " + str(counter),
        (450, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        5
    )

    cv2.imshow("Original Video", frame1)

    # normal speed
    if cv2.waitKey(30) & 0xFF == 13:
        break

cap.release()
cv2.destroyAllWindows()


#opencv
#cv2
#computer_vision
#background_subtraction
#motion_detection
#contours
#object_detection
#image_processing
#video_processing
#numpy