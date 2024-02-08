import cv2
import face_recognition

img1 = face_recognition.load_image_file("C:\\Users\\Selman\\Desktop\\Tasarim-1\\face_recognition\\images\\ben.png")
img1_encoding = face_recognition.face_encodings(img1)[0]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read() 
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640,480))

    #Finding face coordinates 
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations) # encoded the locations taken from each frame

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        #face_encoding and the right, left, lower and upper parts of the face go through face_locations and face_encodings zip combines them
        
        match = face_recognition.compare_faces([img1_encoding], face_encoding) #Compares our normal image and the encodings in the frames

        if True in match:
            cv2.rectangle(frame, (left,top), (right,bottom), (0,255,0), 2)
            cv2.putText(frame, "Selman", (left+6, bottom-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    
    cv2.imshow("Face Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
            

# The face_recognition library was used in these codes and this library is based on the "dlib" library for face recognition.
# It uses dlib's trained face recognition model as the algorithm.
"""
Dlib's face recognition algorithm includes a set of algorithms designed to first detect faces and then represent those faces with feature vectors. Below is an overview explaining the basic steps of Dlib's facial recognition algorithm:

1)Histogram of Oriented Gradients (HOG):
Dlib's face recognition algorithm first extracts Histogram of Oriented Gradients (HOG) features in each region of the image. HOG is a feature vector used to represent important features of an image such as edges, lines, and patterns.

2)Combining HOG Features:
HOG features are extracted in different regions of the image and these feature vectors are combined to obtain an overall feature vector representing a face.

3)Support Vector Machine (SVM):
The resulting feature vectors are trained with a Support Vector Machine (SVM) classifier. SVM is learned to distinguish face and non-face regions.

4)Face Classification:
Using the trained SVM, it is determined whether a face is present in a certain region of an image. This step is used to detect potential facial regions in an image.

5)Face Landmarks:
If a face is detected, the algorithm identifies key points of that face (such as the position of the eyes, nose, and mouth). These facial points can be used for further analysis and recognition.

6)Creating Facial Feature Vector:
The HOG features of the face and facial points are combined to form a feature vector. This feature vector is used to distinguish the face from other faces.

These steps form the basis of Dlib's facial recognition algorithm. This algorithm is designed to provide high accuracy and overall performance. However, as noted, different approaches and models for facial recognition exist, and the preferred method may vary depending on application needs and performance requirements.
"""
           
