#capture stream from video output e.g webcam, camera, and send video frame by frame to server for processing
import cv2,socket,pickle,os
import numpy as np
import time
s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,1000000)
server_ip = "127.0.0.1"
server_port = 6666

cap = cv2.VideoCapture(0)
while True:
	time.sleep(2)
	ret,photo = cap.read()
	cv2.imshow('streaming',photo)
	ret,buffer = cv2.imencode(".jpg",photo,[int(cv2.IMWRITE_JPEG_QUALITY),30])
	x_as_bytes = pickle.dumps(buffer)
	print("sending camera output")
	s.sendto((x_as_bytes),(server_ip,server_port))
	if cv2.waitKey(10)==13: #press enter to stop
		break
cv2.destroyAllWindows()
cap.release()
