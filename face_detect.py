import cv2
import sys
import time

count = 1

#cascPath = sys.argv[1] if len(sys.argv)>1 else '.'
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture('demo2.mp4')

while video_capture.isOpened():

    # Capture frame-by-frameq
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.01,
        minNeighbors=5,
        minSize=(96, 96),
        maxSize=(96, 96),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        crop_img = frame[y: y + h, x: x + w]  # Crop from x, y, w, h -> 100, 200, 300, 400
        cv2.imwrite("C:/Users/BHISHMA/lock_down/Image Detection/images/face" + str(count) + ".png", crop_img)
        count = count + 1
        # Display the resulting frame
    cv2.imshow('GUI', frame)



    cv2.waitKey(1)

video_capture.release()
cv2.destroyAllWindows()
