import cv2 as cv

# Read image from your local file system
# original_image = cv.imread('path/to/your-image.jpg')
video_capture = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
# upper_body_cascade = cv.CascadeClassifier('haarcascade_upperbody.xml')

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    grayscale_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    detected_faces = face_cascade.detectMultiScale(grayscale_image)
    # detected_upper_bodies = upper_body_cascade.detectMultiScale(grayscale_image)

    for (column, row, width, height) in detected_faces:
      cv.rectangle(
        frame,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
    )
    # for (column, row, width, height) in detected_upper_bodies:
    #   cv.rectangle(
    #     frame,
    #     (column, row),
    #     (column + width, row + height),
    #     (255, 0, 0),
    #     2
    # )

    # Display the resulting frame
    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv.destroyAllWindows()