
import cv2
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
faceDetector = cv2.CascadeClassifier('algoritma_wajah.xml')
data_wajah_directory = 'data_wajah'
faceID = input("Masukkan ID wajah: ")
count = 1
while True:
    ret, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu, 1.3, 5)  # 0-2
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        namaFile = "wajah." + str(faceID) + "." + str(count) + ".jpg"
        cv2.imwrite(data_wajah_directory+'/'+namaFile, frame)
        count += 1
    cv2.imshow('Webcamku', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif count > 20:
        break
cam.release()
cv2.destroyAllWindows()
