import cv2
import numpy as np
cap=cv2.VideoCapture(0)

face_detect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detect=cv2.CascadeClassifier('frontalEyes35x16.xml')
noise_detect=cv2.CascadeClassifier('Nose18x15.xml')
mustache=cv2.imread('mustache.png',-1)
glasses=cv2.imread('glasses.png',-1)

while True:
	r,frame=cap.read()
	# check return is false
	if r==False:
		continue

	gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	#face detection using haarcascade_frontalface_default.xml
	faces=face_detect.detectMultiScale(gray_frame,1.2,5)
	frame=cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)
	glasses=cv2.cvtColor(glasses,cv2.COLOR_BGR2BGRA)
	for (x,y,w,h) in faces:
		gray_roi=gray_frame[y:y+w,x:x+h]
		img_roi=frame[y:y+w,x:x+h]
		# eyes detection using frontalEyes35x16.xml
		eyes=eye_detect.detectMultiScale(gray_roi,1.2,5)
		for (ex,ey,ew,eh) in eyes:
			roi_eyes=gray_roi[ey:ey+ew,ex:ex+eh]
			r=eh/float(glasses.shape[1])
			h=int(glasses.shape[0]*r)
			# setting glasses to face
			glasses2=cv2.resize(glasses,(int(eh*1.2),int(h*1.2)))
			gw,gh,gc=glasses2.shape
			for i in range(gw):
				for j in range(gh):
					if glasses2[i,j][3]!=0:
						img_roi[ey+i,ex+j-17]=glasses2[i,j]
		# noise dectection using Nose18x15.xml
		noises=noise_detect.detectMultiScale(gray_roi,1.2,5)
		noise=np.array([noises[0]])
		for (nx,ny,nw,nh) in noise:
			roi_eyes=gray_roi[ny:ny+nw,nx:nx+nh]
			# setting mustache to face
			r=nh/float(mustache.shape[1])
			h=int(mustache.shape[0]*r)
			mustache2=cv2.resize(mustache,(int(nh*1.3),int(h*1.3)))
			gw,gh,gc=mustache2.shape
			for i in range(gw):
				for j in range(gh):
					if mustache2[i,j][3]!=0:
						img_roi[ny+int(nh/2.0)+i,nx+j-5]=mustache2[i,j]



	frame=cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
	cv2.imshow('Video Frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()