import os
import cv2
import numpy as np
from tqdm import tqdm
from rtmlib import Wholebody, draw_skeleton

def main():

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
            try:
                pre_processed_images = []
                for i in range(len(ns)):
                    n, ftype = ns[i]
                    # Original image
                    img_orig = cv2.imread(f'{root}/{label}/{idee}.{n}.{ftype}')

                    keypoints, scores = wholebody(img_orig)

                    # Skeleton image
                    img_spooky = np.zeros(img_orig.shape, dtype=np.uint8)
                    img_spooky = draw_skeleton(img_spooky, keypoints, scores, kpt_thr=0.5)

                    pre_processed_images.append(img_spooky)

                # Added optical flow
                if (len(pre_processed_images) > 1):
                    image_final = opt_flow(pre_processed_images)
                else:
                    image_final = pre_processed_images[0]
            
                cv2.imwrite(f'{dst}/{label}_{idee}.{ftype}', image_final)

            except:
                print("\nSample identification failed. Skipping...\n")

def opt_flow(images):
    # params for ShiTomasi corner detection
    feature_params = dict(  maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
 
    # Parameters for lucas kanade optical flow
    lk_params = dict(   winSize = (15, 15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
 
    # Take first frame and find corners in it
    old_frame = images[0]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    
    for frame in images:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)

        img = cv2.add(frame, mask)
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    return img

if __name__ == "__main__":
    main()
