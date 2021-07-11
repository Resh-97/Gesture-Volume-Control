import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
########################################
wCam, hCam = 640,480
########################################

cam = cv2.VideoCapture(0)
cam.set(3,wCam)
cam.set(4,hCam)

prevTime = 0
currTime = 0

detector = htm.handDetector(detectionConfidence=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
#print(volume.GetVolumeRange())
#-20 -->26 volume
#-5 -->72 volume
#0 -->100 volume
minVol =volRange[0]
maxVol = volRange[1]

while True:
    success, img =cam.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        #print(lmList[4],lmList[8])
        x1,y1 =lmList[4][1],lmList[4][2]
        cv2.circle(img, (x1, y1),15,(255,0,0), cv2.FILLED )

        x2,y2 =lmList[8][1],lmList[8][2]
        cv2.circle(img, (x2, y2), 15,(255,0,0), cv2.FILLED )

        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)

        x_center, y_center = (x1 + x2)//2, (y1 + y2)//2
        cv2.circle(img, (x_center, y_center), 15,(255,0,0), cv2.FILLED )

        ##The idea is change the volume based on the length between the points
        #math.hypot() computes the Eucliden distance: sqrt((x2-x1)^2 + (y2-y1)^2)
        length = math.hypot(x2-x1,y2-y1)
        #print(length)

        # Hand Range : 50 - 300
        # Volume Range : -65 -> 0
        vol = np.interp(length,[20,200],[minVol,maxVol])
        volBar = np.interp(length,[20,200],[400,150])
        volPer = np.interp(length,[20,200],[0,100])
        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length<50:
            cv2.circle(img, (x_center, y_center), 15,(255,0,255), cv2.FILLED )

        cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
        cv2.rectangle(img,(50,int(volBar)),(85,400),(0,255,0),cv2.FILLED)

        cv2.putText(img, f'{int(volPer)}%',(20,430), cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,0),3)


    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv2.putText(img, f'FPS:{int(fps)}',(10,50), cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,0),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
