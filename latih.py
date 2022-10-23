import cv2
import os
import numpy as np
import time
from PIL import Image

# Path for face image database
data_wajah_directory = 'dataset_gabungan'
hasil_latih_directory = 'dataset_gabungan_hasil_latih'
# Create Local Binary Patterns Histograms for face recognization


def getImageLabel(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    faceIDs = []
    for imagePath in imagePaths:
        PILImg = Image.open(imagePath).convert('L')  # convert it to grayscale
        imgNum = np.array(PILImg, 'uint8')
        faceID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = faceDetector.detectMultiScale(imgNum)
        for (x, y, w, h) in faces:
            faceSamples.append(imgNum[y:y+h, x:x+w])
            faceIDs.append(faceID)
    return faceSamples, faceIDs


faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetector = cv2.CascadeClassifier('algoritma_wajah.xml')

print("Mengenali wajah. Tunggu sebentar...")
faces, faceIDs = getImageLabel(data_wajah_directory)
faceRecognizer.train(faces, np.array(faceIDs))

faceRecognizer.write(hasil_latih_directory + '/hasil_latih.yml')
print("Proses pelatihan telah selesai...")
