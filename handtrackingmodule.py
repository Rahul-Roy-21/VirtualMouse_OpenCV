import cv2
import mediapipe as mp
import time
import numpy as np
import math

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionConf=0.5, trackConf=0.5):
        self.mode = mode #static_image_mode
        self.maxHands = maxHands
        self.detection_Conf = detectionConf
        self.track_Conf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode = self.mode,
            max_num_hands = self.maxHands,
            min_detection_confidence = self.detection_Conf,
            min_tracking_confidence = self.track_Conf
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.fingertip_ids = tuple(i for i in range(4,21,4))

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # Checking for Existence of Multiple Hand_Landmarks
        if self.results.multi_hand_landmarks:
            for handlmks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        image = img, 
                        landmark_list = handlmks, 
                        connections = self.mpHands.HAND_CONNECTIONS,
                        landmark_drawing_spec = self.mpDraw.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=5), 
                        connection_drawing_spec = self.mpDraw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                    )

        # Return the Image with Connected Hand Landmarks drawn on it
        return img

    def findPosition(self, img, handIndex=0, draw=True):
        self.lmklist = []
        h,w,c = img.shape
        # Getting the Boundaries of our Hand-Box
        xList, yList = [], []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handIndex]
            for lmid, lmk in enumerate(myHand.landmark):
                cx, cy = int(lmk.x*w), int(lmk.y*h)
                xList.append(cx)
                yList.append(cy)

                self.lmklist.append([lmid, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)

        if len(xList) != 0:
            xmin, xmax, ymin, ymax = min(xList), max(xList), min(yList), max(yList)
        else:
            xmin, xmax, ymin, ymax = 0, 0, 0, 0 # Empty List, Camera Not Yet Processed

        borderbox = {'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax': ymax}
        if draw:
            cv2.rectangle(img, (xmin-20,ymin-20), (xmax+20,ymax+20), (0,255,0), 2)

        return self.lmklist, borderbox

    def fingersUp(self):
        fingers, palmVisible = [], 1

        # Check if The Palm is in Front of Camera?
        if self.lmklist[0][1] > self.lmklist[2][1]:
            palmVisible = 0

        # Thumb
        if self.lmklist[self.fingertip_ids[0]][1] > self.lmklist[self.fingertip_ids[0]-1][1] and palmVisible:
            fingers.append(1)
        elif self.lmklist[self.fingertip_ids[0]][1] < self.lmklist[self.fingertip_ids[0]-1][1] and not palmVisible:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other Fingers
        for id in range(1,5):
            if self.lmklist[self.fingertip_ids[id]][2] < self.lmklist[self.fingertip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=8, t=3):
        x1,y1 = self.lmklist[p1][1:]
        x2,y2 = self.lmklist[p2][1:]
        cx,cy = (x1+x2)//2, (y1+y2)//2
        linepts = [(x1,y1),(x2,y2),(cx,cy)]
        # print(linepts)

        if draw:
            cv2.line(img, (x1,y1), (x2,y2), (255,0,255), t)
            cv2.circle(img, (x1,y1), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (x2,y2), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (cx,cy), r, (0,0,255), cv2.FILLED)
            length = math.hypot(x2-x1, y2-y1)

        return length, img, linepts

if __name__=='__main__':
    pTime, cTime = 0,0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist, bbox = detector.findPosition(img)

        if len(lmlist):
            print(lmlist[4], detector.fingersUp())

        # Calculating Frame Rate(fps)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, f'FPS:{int(fps)}', (10,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        cv2.imshow('IMAGE', img)
        if cv2.waitKey(1) & 0xFF == ord('d'):
            break
    
    cap.release()
    cv2.destroyAllWindows()