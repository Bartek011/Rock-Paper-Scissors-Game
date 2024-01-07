import random
import cv2
import cvzone
import mediapipe as mp
import time
from hand import *

timer = 0
state_result = False
start_game = False
mp_hands = mp.solutions.hands
score = [0,0] # [player, pc]

# Begin capture camera
cap = cv2.VideoCapture(0)
cap.set(3,640) # Set camera width to 640px
cap.set(4,480) # Set camera height to 480px


with mp_hands.Hands(
    static_image_mode = True,
    max_num_hands = 1,
    min_detection_confidence = 0.5) as hands:
        while True:
            img_background = cv2.imread("assets/background.png")

            success, img = cap.read()

            img_scaled = cv2.resize(img, (0,0), None,0.82,0.82)
            img_scaled = img_scaled[:,64:461]

            # Start the game
            if cv2.waitKey(1) == ord('s'):
                start_game = True
                initialTime = time.time()
                state_result = False

            if start_game:
                if state_result is False:
                    timer = time.time() - initialTime
                    cv2.putText(
                        img_background,
                        str(int(4 - timer)),
                        (598,595),
                        fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale = 3,
                        color = (163,73,164),
                        thickness = 7)
                    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    player_move = ""

                    if results.multi_hand_landmarks:
                        hn = results.multi_hand_landmarks[0]
                        ip = hn.landmark[INDEX_FINGER_PIP].y
                        it = hn.landmark[INDEX_FINGER_TIP].y
                        mp = hn.landmark[MIDDLE_FINGER_PIP].y
                        mt = hn.landmark[MIDDLE_FINGER_TIP].y
                        rp = hn.landmark[RING_FINGER_PIP].y
                        rt = hn.landmark[RING_FINGER_TIP].y
                        pp = hn.landmark[PINKY_PIP].y
                        pt = hn.landmark[PINKY_TIP].y

                        if (condition_rock(ip, it, mp, mt, rp, rt, pp, pt)):
                            player_move = "ROCK"
                        elif (condition_paper(ip, it, mp, mt, rp, rt, pp, pt)):
                            player_move = "PAPER"
                        elif (condition_scissors(ip, it, mp, mt, rp, rt, pp, pt)):
                            player_move = "SCISSORS"

                    cv2.rectangle(img_scaled, (0,394-75), (397,394), (192,128,0), -1)

                    text_size, _ = cv2.getTextSize(
                        text=player_move,
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=2,
                        thickness= 2)
                    text_width, text_height = text_size
                    cv2.putText(
                        img=img_scaled,
                        text=player_move,
                        org=((397 - text_width) // 2, 394-10),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=2,
                        color=(0,0,0),
                        thickness=2,
                        lineType=cv2.LINE_AA)

                    # Stop the timer after 3sec
                    if timer > 3:
                        state_result = True
                        timer = 0
                        # Choose pc move
                        pc_move = random.randint(1,3)

                        # Logic of the game
                        if player_move == "ROCK":
                            if pc_move == 2:
                                score[1] += 1
                            elif pc_move == 3:
                                score[0] += 1
                        elif player_move == "PAPER":
                            if pc_move == 1:
                                score[0] += 1
                            elif pc_move == 3:
                                score[1] += 1
                        elif player_move == "SCISSORS":
                            if pc_move == 1:
                                score[1] += 1
                            elif pc_move == 2:
                                score[0] += 1

            img_background[285:679,67:464] = img_scaled

            if state_result is True:
                img_pc = cv2.imread(f'assets/{pc_move}.png', cv2.IMREAD_UNCHANGED)
                img_background = cvzone.overlayPNG(img_background, img_pc, (800, 300))

            cv2.putText(img_background, str(score[0]), (545, 410), cv2.FONT_HERSHEY_TRIPLEX, 2, (164,73,163), 3)
            cv2.putText(img_background, str(score[1]), (665, 410), cv2.FONT_HERSHEY_TRIPLEX, 2, (164, 73, 163), 3)
            cv2.imshow("background", img_background)

            if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()

