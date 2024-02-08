import face_recognition
import cv2

path = "C:\\Users\\Selman\\Desktop\\Tasarim-1\\face_recognition\\images\\terim_muslera.png"
image = cv2.imread(path)

# access the position of the face
faceLocations = face_recognition.face_locations(image)
# print(faceLocations) 

pt1_0 = (255,46)
pt2_0 = (344,136)

pt1_1 = (444,126)
pt2_1 = (534,216)

color = (0,255,0)

cv2.rectangle(image, pt1_0, pt2_0, color)
cv2.rectangle(image, pt1_1, pt2_1, color)

cv2.imshow("Test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()