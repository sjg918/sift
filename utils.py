

import torch
import sift_on_gpu
import cv2
import numpy as np
import time
import numba
import math

#refer : https://learnopencv.com/rotation-matrix-to-euler-angles/

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])


#refer : https://github.com/IIPCVLAB/LCCNet
def quaternion_from_rotation_matrix(matrix):
    if matrix.shape == (4, 4):
        R = matrix[:3, :3]
    elif matrix.shape == (3, 3):
        R = matrix
    else:
        raise TypeError("Not a valid rotation matrix")
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    q = np.zeros(4, dtype=np.float32)
    if tr > 0.:
        S = np.sqrt(tr+1.0) * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q / np.linalg.norm(q)


def rotation_matrix_from_quaternion(q):
    mat = np.zeros((3,3), dtype=np.float32)

    mat[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    mat[0, 1] = 2*q[1]*q[2] - 2*q[3]*q[0]
    mat[0, 2] = 2*q[1]*q[3] + 2*q[2]*q[0]
    mat[1, 0] = 2*q[1]*q[2] + 2*q[3]*q[0]
    mat[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    mat[1, 2] = 2*q[2]*q[3] - 2*q[1]*q[0]
    mat[2, 0] = 2*q[1]*q[3] - 2*q[2]*q[0]
    mat[2, 1] = 2*q[2]*q[3] + 2*q[1]*q[0]
    mat[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    return mat


@numba.jit(nopython=True)
def rot_and_trs_points_kernel(points, newpoints, R, numpoints):
    for i in range(numpoints):
        p = points[i]
        kew = np.dot(R, p.T)
        newpoints[i, 0] = kew[0]
        newpoints[i, 1] = kew[1]
        newpoints[i, 2] = kew[2]
        continue

def rot_and_trs_points(points ,R):
    o = np.ones((points.shape[0], 1), dtype=np.float32)
    points = np.concatenate((points, o), axis=1)
    newpoints = np.zeros((points.shape[0], 3), dtype=np.float32)
    rot_and_trs_points_kernel(points.astype(np.float32), newpoints, R.astype(np.float32), points.shape[0])
    return newpoints


@numba.jit(nopython=True)
def mk_input_right_image_kernel(input_right_image, flattenimg, ucoord, vcoord, numpoints):
    for i in range(numpoints):
        h = vcoord[i]
        w = ucoord[i]
        b = flattenimg[i, 3]
        g = flattenimg[i, 4]
        r = flattenimg[i, 5]
        input_right_image[h, w, 0] = b
        input_right_image[h, w, 1] = g
        input_right_image[h, w, 2] = r

def mk_input_right_image(flattenimg, k_cam3, h, w):
    f_u = k_cam3[0]
    f_v = k_cam3[1]
    c_u = k_cam3[2]
    c_v = k_cam3[3]

    flattenimg[:, 0] = flattenimg[:, 0] * f_u + c_u
    flattenimg[:, 1] = flattenimg[:, 1] * f_v + c_v

    mask = (flattenimg[:, 0] > 0) * (flattenimg[:, 0] < w-1) * (flattenimg[:, 1] > 0) * (flattenimg[:, 1] < h-1)
    flattenimg = flattenimg[mask]

    numpoints = flattenimg.shape[0]
    input_right_image = np.zeros((h, w, 3), dtype=np.uint8)
    ucoord = flattenimg[:, 0].astype(np.int32)
    vcoord = flattenimg[:, 1].astype(np.int32)
    mk_input_right_image_kernel(input_right_image, flattenimg, ucoord, vcoord, numpoints)
    return input_right_image

if __name__ == '__main__':    
    Limg = cv2.imread('left.png')
    Rimg = cv2.imread('right.png')
    h, w, _ = Limg.shape

    maxrot = 2.5
    rotx = np.random.uniform(-maxrot, maxrot) * (3.141592 / 180.0)
    roty = np.random.uniform(-maxrot, maxrot) * (3.141592 / 180.0)
    rotz = np.random.uniform(-maxrot, maxrot) * (3.141592 / 180.0)

    miscalibrot = eulerAnglesToRotationMatrix([rotx, roty, rotz])
    miscalibRT = np.zeros((4, 4), dtype=np.float32)
    miscalibRT[:3, :3] = miscalibrot[:3,:3]
    miscalibRT[0, 3] = 0
    miscalibRT[1, 3] = 0
    miscalibRT[2, 3] = 0
    miscalibRT[3, 3] = 1

    c, r = np.meshgrid(np.arange(w), np.arange(h))
    ones = np.ones((h, w), dtype=np.float32)
    points = np.stack([c, r, ones])
    points = points.reshape((3, -1))
    points = points.T

    f_u = 7.070912e+02
    f_v = 7.070912e+02
    c_u = 6.018873e+02
    c_v = 1.831104e+02
    k_cam3 = [f_u, f_v, c_u, c_v]

    points[:, 0] = (points[:, 0] - c_u) / f_u
    points[:, 1] = (points[:, 1] - c_v) / f_v
    normalized_points = points

    flatten_right_img = Rimg.reshape(-1, 3)
    normalized_points = rot_and_trs_points(normalized_points, np.linalg.inv(miscalibRT))
    flattenimg = np.concatenate((normalized_points, flatten_right_img), axis=-1)

    input_right_image = mk_input_right_image(flattenimg, k_cam3, h, w)


    Limg = cv2.cvtColor(Limg, cv2.COLOR_BGR2GRAY)
    Rimg = cv2.cvtColor(input_right_image, cv2.COLOR_BGR2GRAY)

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
    newimg = np.zeros((h, w*2, 3)).astype(np.uint8)
    newimg[:, :w, :] = Limg
    newimg[:, w:, :] = input_right_image

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