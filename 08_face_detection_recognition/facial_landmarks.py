import face_recognition
from PIL import Image, ImageDraw 

path = "C:\\Users\\Selman\\Desktop\\Tasarim-1\\face_recognition\\images\\terim_muslera.png"

image = face_recognition.load_image_file(path)


landmarks = face_recognition.face_landmarks(image)

PILImage = Image.fromarray(image)
d = ImageDraw.Draw(PILImage)

for landmark in landmarks:
    for feature in landmark.keys():
        d.line(landmark[feature], width=3)

PILImage.show()