import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from rtmlib import Wholebody, draw_skeleton

root = './training'
dst = './processed'
labels = sorted(os.listdir(root), key=int)
print(*labels)

device = 'cuda'  # cpu, cuda
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

wholebody = Wholebody(to_openpose=openpose_skeleton,
            mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
            backend=backend, device=device)


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
        image = cv2.imread(f'{root}/{label}/{idee}.{n}.{ftype}')

        keypoints, scores = wholebody(image)

        image = np.zeros(image.shape, dtype=np.uint8)

        img_ready = draw_skeleton(image, keypoints, scores, kpt_thr=0.5)
        cv2.imwrite(f'{dst}/{label}_{idee}.{ftype}', img_ready)