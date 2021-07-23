import cv2 as cv
import dlib
import numpy as np

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv.fillConvexPoly(mask, points, 255)
    return mask

def contouring(thresh, mid, img, right=False):
    cnts, _ = cv.findContours(thresh, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv.contourArea)
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv.circle(img, (cx, cy), 4, (0, 0, 255), 2)
    except:
        pass







face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]


video_capture = cv.VideoCapture(0)

ret, frame = video_capture.read()
thresh = frame.copy()

cv.namedWindow('image')
kernel = np.ones((9,9), np.uint8)

def nothing(x):
    pass

cv.createTrackbar('threshold', 'image', 0, 255, nothing)

while True:
    ret, frame = video_capture.read()

    hog_face_detector = dlib.get_frontal_face_detector()

    #dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    colors = [(0, 255, 255),(0, 0, 255),(255, 255, 0)]
    color=colors[0]
    #_, frame = video_capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = predictor(gray, face)
        if(abs(face_landmarks.part(61).y - face_landmarks.part(67).y) >= 4):
            color = colors[2]
        else:
            color = colors[0]
        for n in range(49, 68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv.circle(frame, (x, y), 1,color , 1)


    #cv.imshow("Face Landmarks", frame)

    key = cv.waitKey(1)
    if key == 27:
        break
    grayscale_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    detected_faces = face_cascade.detectMultiScale(grayscale_image)

    rects = detector(grayscale_image, 1) # rects contains all the faces detected

    for rect in rects:

        shape = predictor(grayscale_image, rect)
        shape = shape_to_np(shape)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv.dilate(mask, kernel, 5)
        eyes = cv.bitwise_and(frame, frame, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv.cvtColor(eyes, cv.COLOR_BGR2GRAY)
        threshold = cv.getTrackbarPos('threshold', 'image')
        _, thresh = cv.threshold(eyes_gray, threshold, 255, cv.THRESH_BINARY)
        thresh = cv.erode(thresh, None, iterations=2) #1
        thresh = cv.dilate(thresh, None, iterations=4) #2
        thresh = cv.medianBlur(thresh, 3) #3
        thresh = cv.bitwise_not(thresh)
        contouring(thresh[:, 0:mid], mid, frame)
        contouring(thresh[:, mid:], mid, frame, True)

    for (column, row, width, height) in detected_faces:
        cv.rectangle(
            frame,
            (column, row),
            (column + width, row + height),
            (0, 255, 0),
            2
        )
        cv.putText(
            frame,
            'rosto',
            (column, row - 10), 
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv.imshow('olhos', frame)
    cv.imshow("image", thresh)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()