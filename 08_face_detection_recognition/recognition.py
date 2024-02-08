import cv2
import face_recognition

image = cv2.imread("C:\\Users\\Selman\\Desktop\\Tasarim-1\\face_recognition\\images\\marlon_brando_test.png")

path = "C:\\Users\\Selman\\Desktop\\Tasarim-1\\face_recognition\\images\\marlon_brando.png"

brandoImage = face_recognition.load_image_file(path) #We read the image with the face_recognition library and now we will produce certain values.
brandoImageEncoding = face_recognition.face_encodings(brandoImage)[0]
#It created special issues featuring Brando's artwork. If there was more than one face, it could be [1]

testPath = "C:\\Users\\Selman\\Desktop\\Tasarim-1\\face_recognition\\images\\marlon_brando_test.png"
testImage = face_recognition.load_image_file(testPath)
faceLoc = face_recognition.face_locations(testImage)
faceEncoding = face_recognition.face_encodings(testImage, faceLoc)

#print(faceLoc) (33,589,718,204)

topLeftX = 204
topLeftY = 33
bottomRightX = 700
bottomRightY = 718

matchedFaces = face_recognition.compare_faces(brandoImageEncoding, faceEncoding)

if True in matchedFaces:
    cv2.rectangle(image, (topLeftX,topLeftY), (bottomRightX,bottomRightY), (0,255,0), 2)
    cv2.putText(image, "Marlon Brando", (topLeftX,topLeftY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    cv2.rectangle(image, (topLeftX,topLeftY), (bottomRightX,bottomRightY), (0,255,0), 2)
    cv2.putText(image, "Unknown", (topLeftX,topLeftY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    cv2.imshow("Face Detection", image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

