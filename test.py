from crop_fingertip import *
import cv2

img_dir = "test_img.jpg"

result = get_fingertip(img_dir)

for k in result.keys():
    tip = result[k]
    for t in tip:
        img = t[0]
        # print(tip)



        cv2.imshow("tip", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
