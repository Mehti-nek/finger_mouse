import cv2
import time
import numpy as np
import mediapipe as mp
import mouse

mp_handmesh = mp.solutions.hands
mp_hands = mp_handmesh.Hands(max_num_hands=1,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

def map_range(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min

pTime = 0
cTime = 0

# drawing tools
tools = cv2.imread("tools.png")
tools = tools.astype('uint8')

cap = cv2.VideoCapture(0)

x = 0
y = 0

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(rgb)
    #print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            # mp_drawing.draw_landmarks(frame, handLms, mp_handmesh.HAND_CONNECTIONS)
            x1, y1 = int(handLms.landmark[4].x*640), int(handLms.landmark[4].y*480)
            cv2.circle(frame, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
            x2, y2 = int(handLms.landmark[8].x*640), int(handLms.landmark[8].y*480)
            cv2.circle(frame, (x2, y2), 5, (0, 255, 0), cv2.FILLED)
            x3, y3 = int(handLms.landmark[12].x*640), int(handLms.landmark[12].y*480)
            cv2.circle(frame, (x3, y3), 5, (0, 0, 255), cv2.FILLED)

            # print(f"{x1-x2} : {y1-y2}")
            x = map_range(x1, 0, 640, 0, 1920)
            y = map_range(y1, 0, 480, 0, 1080)

            mouse.move(x, y, True)
            # print(f"{x} : {y}")

            if abs(x1-x2) < 20 and abs(y1-y2) < 20:
                mouse.click('left')
                # print('touch')

            if abs(x1-x3) < 20 and abs(y1-y3) < 20:
                mouse.click('right')
                # print('touch')

            
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_PLAIN,
                2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Image", frame)

    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break