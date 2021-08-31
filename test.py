import os
import cv2
import glob
import os
import numpy as np
from util.preprocessing import *
from util.crop_fingertip import *


def main():
    # img_dir = sys.argv[1]
    img_dir = "./내손을잡아요/modified.png"

    finger_tips = get_fingertip(img_dir)

    if os.path.isfile(img_dir):
        print("here")
        
        original_img = cv2.imread(img_dir)

    print(finger_tips)
    for name in finger_tips.keys():
        items = finger_tips[name]
        count = 0

        if os.path.isdir(img_dir):
            original_img = cv2.imread(os.path.join(img_dir, name))

        original_img = cv2.flip(original_img, 1)

        for item in items:
            img, top_left, down_right = item
            img = cv2.resize(img, (256, 256))

            finger, mask = get_mask(img, 0.9)
            edge = extract_edge(finger)
            enhanced = Enhancement(edge, mask)

            # img_name = name.split(".")[0] + str(count) + ".png"
            output_dir = "./test/output/"
            # count += 1
            img_name = "img.png"

            test_dir = "./test"

            cv2.imwrite(os.path.join("./test/", img_name), img)
            cv2.imwrite("./test/mask.png", mask)
            cv2.imwrite("./test/edge.png", enhanced)

            os.system(
                "python test.py --model 2 --checkpoints ./edge-connect/checkpoints  --input {test_dir}/{img_name}  --mask {test_dir}/mask.png --edge {test_dir}/edge.png  --output {output_dir}".format(
                    output_dir=output_dir, img_name=img_name, test_dir=test_dir
                )
            )

            changed = cv2.imread(os.path.join(output_dir, img_name))

            changed = cv2.resize(
                changed, (down_right[1] - top_left[1], down_right[0] - top_left[0])
            )

            original_img[
                top_left[1] : down_right[1], top_left[0] : down_right[0]
            ] = changed

        cv2.imwrite(
            "{test_dir}/{name}_modified.jpg".format(
                name=name.split(".")[0], test_dir=test_dir
            ),
            original_img,
        )


if __name__ == "__main__":
    main()
