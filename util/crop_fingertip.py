import cv2
import math
import numpy as np
import os
import json
import sys
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3
)


def get_coor(landmark, i):
    coor = np.array([landmark[i].x, landmark[i].y])
    return coor


def dist(c1, c2):
    return np.sqrt(np.square(c1[0] - c2[0]) + np.square(c1[1] - c2[1]))


def check_linear(landmark, i):
    c1 = get_coor(landmark, i)
    c2 = get_coor(landmark, i - 1)
    c3 = get_coor(landmark, i - 2)
    c4 = get_coor(landmark, i - 3)
    c5 = get_coor(landmark, 0)
    v1 = c1 - c2
    v2 = c2 - c3
    v3 = c3 - c4
    v4 = c4 - c5
    v5 = c1 - c5
    # if np.dot(v1, v2) >0 and np.dot(v2, v3) >0 and np.dot(v3, v4) >0 :
    if dist(c1, c5) > dist(c2, c5):
        return True
    else:
        return False


def get_fingertip(base_dir, save_dir=None):
    if save_dir == None:
        save_dir = os.path.join(base_dir, "result")

    result = dict()

    if os.path.isdir(base_dir):
        img_list = os.listdir(base_dir)
        img_list = [
            os.path.join(base_dir, i)
            for i in img_list
            if os.path.isfile(os.path.join(base_dir, i))
        ]
        images = {dir.split("/")[-1]: cv2.imread(dir) for dir in img_list}
    else:
        images = {base_dir.split("/")[-1]: cv2.imread(base_dir)}
    for name, image in images.items():
        # Convert the BGR image to RGB, flip the image around y-axis for correct
        # handedness output and process it with MediaPipe Hands.

        try:
            results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
        except:
            continue

        if not results.multi_hand_landmarks:
            continue

        image_hight, image_width, _ = image.shape
        annotated_image = cv2.flip(image.copy(), 1)

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):

            HAND = results.multi_handedness[i].classification[0].label
            print(name, HAND)
            landmarks = [4, 8, 12, 16, 20]

            x_var = (
                abs(hand_landmarks.landmark[7].x - hand_landmarks.landmark[8].x)
                * image_width
            )
            y_var = (
                abs(hand_landmarks.landmark[7].y - hand_landmarks.landmark[8].y)
                * image_hight
            )
            pm = int(max(x_var, y_var))

            for l in landmarks:
                if not check_linear(hand_landmarks.landmark, l):
                    print("Failed : ", l)
                    continue
                X = int(hand_landmarks.landmark[l].x * image_width)
                Y = int(hand_landmarks.landmark[l].y * image_hight)
                down_right = [X + pm, Y + pm]
                top_left = [X - pm, Y - pm]

                # index_FT = cv2.flip(image,1 )[top_left[1]:down_right[1], top_left[0]:down_right[0]]
                index_FT = cv2.flip(image, 1)[Y - pm : Y + pm, X - pm : X + pm]
                tip = index_FT
                # tip = cv2.resize(index_FT, (256,256))
                # cv2.imwrite(os.path.join(save_dir,  str(l)+name+ "_"+str() + "_"), tip)

                try:
                    result[name].append((tip, top_left, down_right))
                except:
                    # print(result)
                    result[name] = [(tip, top_left, down_right)]
    return result


# if __name__ == "__main__":
# main()

