

import torch
import sift_on_gpu
import cv2
import numpy as np
import time

if __name__ == '__main__':    
    Limg = cv2.imread('left.png')
    Rimg = cv2.imread('right.png')
    h, w, _ = Limg.shape

    Limg = cv2.cvtColor(Limg, cv2.COLOR_BGR2GRAY)
    Rimg = cv2.cvtColor(Rimg, cv2.COLOR_BGR2GRAY)

    Limg = torch.from_numpy(Limg).to(dtype=torch.float32, device='cuda:0')
    Rimg = torch.from_numpy(Rimg).to(dtype=torch.float32, device='cuda:0')
    num_matched_pts = torch.ones((1), dtype=torch.int32)

    # cold start
    matched_pts = sift_on_gpu.sift(Limg, Rimg, num_matched_pts, w, h, 11)

    # timelist = []
    # for i in range(1000):
    #     starttime = time.time()
    #     matched_pts = sift_on_gpu.sift(Limg, Rimg, num_matched_pts, w, h, 11)
    #     timelist.append(time.time() - starttime)

    matched_pts = matched_pts.view(-1, 4)
    matched_pts = matched_pts[:num_matched_pts[0], :]
    
    Limg = cv2.imread('left.png')
    Rimg = cv2.imread('right.png')
    newimg = np.zeros((h, w*2, 3)).astype(np.uint8)
    newimg[:, :w, :] = Limg
    newimg[:, w:, :] = Rimg

    for i in matched_pts:
        x1, y1, x2, y2 = i
        x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
        cv2.line(newimg, (x1,y1), (w+x2,y2), color=(0, 255, 0), thickness=1)
        continue
    for i in matched_pts:
        x1, y1, x2, y2 = i
        x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
        cv2.circle(newimg, (x1,y1), radius=2, color=(0, 0, 255), thickness=-1)
        cv2.circle(newimg, (w+x2,y2), radius=2, color=(0, 0,255), thickness=-1)
        continue
    cv2.imwrite('hi.png', newimg)
    #print(sum(timelist) / len(timelist))
