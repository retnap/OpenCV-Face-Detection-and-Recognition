import face_recognition
import cv2 

path = "C:\\Users\\Selman\\Desktop\\Tasarim-1\\face_recognition\\videos\\faces.mp4"
cap = cv2.VideoCapture(path)

color = (255,0,0)

while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame, (640,480))

    faceLocations = face_recognition.face_locations(frame)

    for index, faceLoc in enumerate(faceLocations): # With enumerate it will keep the coordinates in faceLocations as an index
        #faceLoc variable will hold the coordinates where the face is located
        topLeftY, bottomRightX, bottomRightY, topLeftX = faceLoc

        pt1=(topLeftX,topLeftY)
        pt2 =(bottomRightX, bottomRightY)

        cv2.rectangle(frame, pt1, pt2, color)

    cv2.imshow("Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
   

   