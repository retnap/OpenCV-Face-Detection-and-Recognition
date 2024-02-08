import face_recognition
import cv2 

cap = cv2.VideoCapture(0)

color = (0,255,0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    faceLocations = face_recognition.face_locations(frame)

    for index, faceLoc in enumerate(faceLocations):
        topLeftY, bottomRightX, bottomRightY, topLeftX = faceLoc
        pt1 = (topLeftX, topLeftY)
        pt2 = (bottomRightX, bottomRightY)

        cv2.rectangle(frame, pt1, pt2, color)

        cv2.imshow("Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()