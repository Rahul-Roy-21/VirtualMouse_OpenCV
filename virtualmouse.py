import cv2
import numpy as np
import time
import handtrackingmodule as htm
import autopy

#------ Custom-Settings ------------------
wCam, hCam = 640, 480
keyClose = ['d','q']
frameRed = 100
smoothening = 5

wScr, hScr = autopy.screen.size()
# print(wScr, hScr)
#-------------------------------------

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
plocX, plocY = 0,0
clocX,clocY = 0,0

clicked = False
# Prevent multi-click when both fingers r up together for consecutive frames
# CLICKING functionality(deactivated after a click) re-activates when middle,ring and little fingers are DOWN together.. 

detector = htm.HandDetector(maxHands=1)

while True:
    
    # 1. Detect Hand-Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Detect the Index and Middle finger tips
    if len(lmList):
        xi, yi = lmList[8][1:]
        xm, ym = lmList[12][1:]
        # print(f'INDEX: {(xi, yi)}, MIDDLE: {(xm, ym)}')

        # 3. Check if the Fingertips are Up
        fingers = detector.fingersUp()
        # print(fingers)
        if not 1 in fingers[2:]:
            clicked = False

        cv2.rectangle(img, (frameRed,frameRed), (wCam - frameRed, hCam - frameRed), (255,0,255), 2)

        # 4. If Index Finger up -> MOUSE MOVING MODE
        if fingers[1] == 1 and fingers[2] == 0:

            # 5. Convert Cooridnates -> Map the Entire Screen to a Finger-trackable Zone(Rectangle)
            xii = np.interp(xi, (frameRed, wCam-frameRed), (0, wScr))
            yii = np.interp(yi, (frameRed, hCam-frameRed), (0, hScr))

            # 6. Smoothen Values to prevent Flickering of Mouse Cursor away from our 'Click' target.
            clocX = plocX +(xii-plocX)/smoothening
            clocY = plocY +(yii-plocY)/smoothening

            # 7. Move Mouse according to Indexfinger tip location in Current Frame
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (xi, yi), 15, (255,0,255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 8. If Both Index & Middle Fingers UP -> MOUSE CLICKING MODE
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, linepts = detector.findDistance(8, 12, img)
            # print(length)

            if length < 35 and not clicked:
                if fingers[3] and fingers[4]:
                    # 9. RIGHT-CLICK -> LEFT-FINGER + Ring and Ping Fingers also up.
                    cv2.circle(img, linepts[2], 13, (0,255,255), cv2.FILLED)
                    autopy.mouse.click(autopy.mouse.Button.RIGHT)
                else:
                    # 10. LEFT-CLICK -> Index & Middle Finger joined
                    cv2.circle(img, linepts[2], 13, (0,255,0), cv2.FILLED)
                    autopy.mouse.click(autopy.mouse.Button.LEFT)
                clicked = True

        
    # 11. Frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS={int(fps)}', (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3)

    # 12. Show Image..
    cv2.imshow('IMAGE', img)
    if cv2.waitKey(1) & 0xFF in list(map(ord, keyClose)):
        break

cap.release()    
cv2.destroyAllWindows()