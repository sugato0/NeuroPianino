import subprocess
from threading import Thread

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import  model_from_json



class HandDetector:

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.fingers = []
        self.lmList = []



    def findHands(self, img, draw=True,flipType = True):


        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(imgRGB)

        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []

                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py])

                myHand["lmList"] = mylmList

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Right"
                    else:
                        myHand["type"] = "Left"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)
                if draw:
                    self.mpDraw.draw_landmarks(imgRGB, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        if draw:
            return allHands, imgRGB
        else:
            return allHands
detector = HandDetector(detectionCon=0.8, maxHands=2)
def crop_res_img(image):

    coef: int
    crop_image: image
    #make square for picture

    if np.shape(image)[0] > np.shape(image)[1]:

        coef = np.shape(image)[0] - np.shape(image)[1]
        if coef % 2 != 0:
            coef -= 1
        #print(coef)
        crop_image = image[int(coef/2):int(np.shape(image)[0] - coef/2)]
    elif np.shape(image)[0] < np.shape(image)[1]:
        coef = np.shape(image)[1] - np.shape(image)[0]
        if coef % 2 != 0:
            coef -= 1
        #print(coef)
        crop_image = image[:, int(coef/2):int(np.shape(image)[1] - coef/2):]
    else:

        crop_image = image

    res_image = cv2.resize(crop_image,(100,100))
    return res_image

#with mediapipe and function crop_res_img process our photo



def RealTimeProcessing():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=2)

    word = ""
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    while True:
        # Get image frame
        global mylmList
        success, img = cap.read()
        ret, fr = cap.read()

        if not success:
            pass
        # size = np.shape(fr)[0],np.shape(fr)[1]
        fr = crop_res_img(fr)


        hands, fr = detector.findHands(fr)

        if hands:
            # Hand 1
            hand1 = hands[0]

            print(hands)
            lmList1 = np.array([hand1["lmList"]])
            # test predictions

            prediction = loaded_model.predict(lmList1)
            print(prediction)
            with open("SongNumbers.txt", "w") as f:
                if prediction[0][0] > 0.5:
                    word = "0"



                elif prediction[0][1] > 0.5:
                    word = "1"

                elif prediction[0][2] > 0.5:
                    word = "2"

                elif prediction[0][3] > 0.5:
                    word = "3"

                elif prediction[0][4] > 0.5:
                    word = "4"

                elif prediction[0][5] > 0.5:
                    word = "5"

                elif prediction[0][6] > 0.5:
                    word = "6"

                elif prediction[0][7] > 0.5:
                    word = "7"

                elif prediction[0][8] > 0.5:
                    word = "10"

                elif prediction[0][9] > 0.5:
                    word = "9"
                else:
                    pass
                f.write(word)
                subprocess.Popen(args=["start", "pyw", "Player.pyw"], shell=True, stdin=subprocess.PIPE)

        #cv2.putText(img, word, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow("test", fr)
        cv2.imshow("main", img)
        cv2.waitKey(1)
RealTimeProcessing()