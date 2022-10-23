import cv2
import os
import numpy as np

data_wajah_directory = 'dataset_gabungan'
hasil_latih_directory = 'dataset_gabungan_hasil_latih'
cam = cv2.VideoCapture(0)
cam.set(3, 1080)  # Width
cam.set(4, 720)  # Height
faceDetector = cv2.CascadeClassifier('algoritma_wajah.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

faceRecognizer.read(hasil_latih_directory + '/hasil_latih.yml')
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['Tidak diketahui', 'Asian', 'Caucasian', 'Indian', 'Negroid']

minWidth = 0.1*cam.get(3)
minHeight = 0.1*cam.get(4)

while True:
    retV, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Flip camera vertically
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(
        abuAbu, 1.2, 5, minSize=(int(minWidth), int(minHeight)))
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
        id, confidence = faceRecognizer.predict(abuAbu[y:y+h, x:x+w])
        if (confidence <= 90):
            nameID = names[id]
            confidenceText = "{0}%".format(round(150 - confidence))
        else:
            nameID = names[0]
            confidenceText = "{0}%".format(round(150 - confidence))
        cv2.putText(frame, str(nameID), (x+5, y-5), font, 1, (0, 255, 0), 3)
        cv2.putText(frame, str(confidenceText),
                    (x+5, y+h-5), font, 1, (0, 255, 0), 2)
    cv2.imshow('Pengenalan Wajah', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
