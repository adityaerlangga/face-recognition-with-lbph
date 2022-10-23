# Face Recognition with Webcam

import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Width
cam.set(4, 480)  # Height

faceDetector = cv2.CascadeClassifier('algoritma_wajah.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(
        abuAbu, 1.3, 5)  # frame, scaleFactor, minNeighbors
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Webcamku', frame)
    # cv2.imshow('Webcamku 2', abuAbu)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cam.release()
cv2.destroyAllWindows()
