import cv2
import dlib
def mounth():
    cap = cv2.VideoCapture(0)

    hog_face_detector = dlib.get_frontal_face_detector()

    dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    colors = [(0, 255, 255),(0, 0, 255),(255, 255, 0)]
    color=colors[0]
    while True:
        _, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = hog_face_detector(gray)
        for face in faces:

            face_landmarks = dlib_facelandmark(gray, face)
            if(abs(face_landmarks.part(61).y - face_landmarks.part(67).y) >= 4):
                color = colors[2]
            else:
                color = colors[0]
            for n in range(49, 68):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                cv2.circle(frame, (x, y), 1,color , 1)


        cv2.imshow("Face Landmarks", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
#cap.release()
#cv2.destroyAllWindows()