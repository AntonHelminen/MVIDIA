import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from rtmlib import Wholebody, draw_skeleton


def main():

    pygame.init()
    pygame.mixer.init()
    sound = pygame.mixer.Sound('sp.mp3')
    sound.set_volume(0.1)

    root = './training'
    #root = './small_training'

    if os.path. exists("training.txt"):
        os. remove("training.txt")

    savefile = open("training.txt", "a", encoding="utf-8")
    labels = sorted(os.listdir(root), key=int)
    print(*labels)

    for label in tqdm(labels[:]):
        files = os.listdir(f'{root}/{label}')

                   
        info = [file.split('.') for file in files]
        sets = {}

        for idee, n, itype in info:
            if idee not in sets:
                sets[idee] = [(n, itype)]
            else:
                sets[idee].append((n, itype))
        for idee, ns in sets.items():
            ns = sorted(ns, key=lambda x: int(x[0]))
            n, ftype = ns[len(ns) // 2]
            image = cv.imread(f'{root}/{label}/{idee}.{n}.{ftype}')
            keypoints = skeleton(image)
            x = keypoints[:,0]
            y = keypoints[:,1]
            for i in range(len(x)):
                savefile.write(f'{x[i]};{y[i]};')
            savefile.write(f'{label}\n')
            print(label)
            #plt.plot(x,y,'k.')
            #plt.show()
            # print(f'{label}, {idee}, {n}')
            # cv.imshow("im",image)
            # cv.waitKey(0)


    savefile.close()
    sound.play()
    time.sleep(27)
    
    print("Kiitos ohjelman käytöstä.")

    return 0

def skeleton(img):
    device = 'cuda'  # cpu, cuda
    backend = 'onnxruntime'  # opencv, onnxruntime, openvino

    openpose_skeleton = False  # True for openpose-style, False for mmpose-style

    wholebody = Wholebody(to_openpose=openpose_skeleton,
                      mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend, device=device)

    keypoints, scores = wholebody(img)

    # if you want to use black background instead of original image,
    # img_show = np.zeros(img_show.shape, dtype=np.uint8)

    # img_show = draw_skeleton(img, keypoints, scores, kpt_thr=0.5)


    return keypoints[0]

if __name__ == "__main__":
    main()
