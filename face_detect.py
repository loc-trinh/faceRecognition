import cv2, glob

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_face(filename):
	image = cv2.imread(filename)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

	if len(faces) == 0:
		return None

	(max_x,max_y,max_w,max_h) = (0,0,0,0)
	for (x,y,w,h) in faces:
		if (w * h) > (max_w * max_h):
			(max_x,max_y,max_w,max_h) = (x,y,w,h)

	cropped = gray[max_y:max_y+max_h, max_x:max_x+max_w]
	resized = cv2.resize(cropped, (100,100))
	return resized.reshape((10000,))