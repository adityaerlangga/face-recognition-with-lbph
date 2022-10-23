# Face Recognition with Webcam

import cv2

cam = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('Webcamku', frame)
    cv2.imshow('Webcamku 2', abuAbu)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cam.release()
cv2.destroyAllWindows()
