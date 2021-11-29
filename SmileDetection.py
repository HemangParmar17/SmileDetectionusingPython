import cv2

#Haarcascade algorithm for Face detection
face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Haarcascade algorithm for Smile detection
smile_detection = cv2.CascadeClassifier('haarcascade_smile.xml')

#Video capture
webcam = cv2.VideoCapture(0)

while True:
    #Read webcam
    read_successful, frame = webcam.read()

    if not read_successful:
        break

    #Convert frame to grayscale
    grayscale_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Find the coordinates of Face
    face_coordinates = face_detection.detectMultiScale(grayscale_frame)

    #Draw rectangle on face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #Find out the SubFace
        the_face = (x, y, w, h)
        the_face = frame[y:y+h, x:x+w]

        #Convert frame to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        #Find the coordinates of Smile
        smile_coordinates = smile_detection.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

        #Draw rectangle around Smile
        #for (x_, y_, w_, h_) in smile_coordinates:
            #cv2.rectangle(the_face, (x_, y_),(x_+w_, y_+h_),(0, 0, 255), 2)

        #Put text if there is a smile
        if len(smile_coordinates)>0:
            cv2.putText(frame, 'Smiling',(x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0, 0, 255) )

    #Display
    cv2.imshow('Smile?', frame)
    key = cv2.waitKey(1)

    #Quit if key q or Q is pressed
    if key == 81 or key == 113:
        break

#Cleanup
webcam.release()
cv2.destroyAllWindows()

print("Smile Detection")
