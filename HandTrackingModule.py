import cv2
import mediapipe as mp
import time
cam = cv2.VideoCapture(0)

class handDetector():

    def __init__(self, mode = False, maxHands = 2, detectionConfidence = 0.5, trackConfidence = 0.5 ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mpHands = mp.solutions.hands
        ## static_image_mode if False will track only when the confidence falls.
        ## True is for continuour tracking but this slows down the processing.
        ## Using default values of the Hands class
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionConfidence,self.trackConfidence )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img, draw =True):
        imgRGB  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Original Images are BGR
        self.results = self.hands.process(imgRGB)  #Processes RGB and returns two fields: "multi_hand_landmarks" & "multi_handedness"
        #print(results.multi_hand_landmarks)

        ##Checking if there are multiple hands and extracting each of them
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) #Draws the keypts & connections
        return img

    def findPosition(self,img,handNumber=0, draw=True):
        lmList =[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]
            for id, lm in enumerate(myHand.landmark): ##lms is the ratio of height and width coord.
                #print(id,lm)
                height, width, channel = img.shape
                x_center, y_center = int(lm.x * width), int(lm.y * height)
                lmList.append([id, x_center, y_center])
                #print(id, x_center, y_center)
                # if id ==4:
                #     cv2.circle(img, (x_center, y_center), 15,(255,0,255), cv2.FILLED )
                if draw:
                     cv2.circle(img, (x_center, y_center), 7,(255,0,0), cv2.FILLED )

        return lmList


def main():
    prevTime = 0
    currTime = 0
    detector = handDetector()
    while True:
        success, img = cam.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0 :
            print(lmList[4]) # Prints the specific landmark position of the defined hand.

        currTime = time.time()
        fps = 1/(currTime-prevTime)
        prevTime = currTime

        cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255),3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__== '__main__':
    main()
