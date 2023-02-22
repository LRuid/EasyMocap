'''
 * @ Date: 2020-09-14 11:01:52
 * @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-04-13 20:31:34
  @ FilePath: /EasyMocapRelease/media/qing/Project/mirror/EasyMocap/easymocap/mytools/reconstruction.py
'''

import numpy as np

def solveZ(A):
    u, s, v = np.linalg.svd(A)
    X = v[-1, :]
    X = X / X[3]
    return X[:3]

def projectN3(kpts3d, Pall):#参考https://blog.csdn.net/weixin_49804978/article/details/121922128
    # kpts3d: (N, 3)
    nViews = len(Pall)
    kp3d = np.hstack((kpts3d[:, :3], np.ones((kpts3d.shape[0], 1))))#kpts3d的最后一维是置信度，这里把他全部变成1
    kp2ds = []#4个相机的投影
    for nv in range(nViews):
        kp2d = Pall[nv] @ kp3d.T
        kp2d[:2, :] /= kp2d[2:, :]#齐次的像素坐标除以第三维等于投影的像素坐标
        kp2ds.append(kp2d.T[None, :, :])
    kp2ds = np.vstack(kp2ds)#堆叠4个视角的投影坐标，(4,25,3)==》(4,25,2+1)其中1代表深度s
    kp2ds[..., -1] = kp2ds[..., -1] * (kpts3d[None, :, -1] > 0.)#判断3d点是否存在，不存在的情况下深度为0
    return kp2ds#输出堆叠4个视角的投影坐标，(4,25,3)==》(4,25,2+1)其中1代表深度s

def simple_reprojection_error(kpts1, kpts1_proj):
    # (N, 3)
    error = np.mean((kpts1[:, :2] - kpts1_proj[:, :2])**2)
    return error
    
def simple_triangulate(kpts, Pall):
    # kpts: (nViews, 3)
    # Pall: (nViews, 3, 4)
    #   return: kpts3d(3,), conf: float
    nViews = len(kpts)
    A = np.zeros((nViews*2, 4), dtype=np.float)
    result = np.zeros(4)
    result[3] = kpts[:, 2].sum()/(kpts[:, 2]>0).sum()
    for i in range(nViews):
        P = Pall[i]
        A[i*2, :] = kpts[i, 2]*(kpts[i, 0]*P[2:3,:] - P[0:1,:])
        A[i*2 + 1, :] = kpts[i, 2]*(kpts[i, 1]*P[2:3,:] - P[1:2,:])
    result[:3] = solveZ(A)

    return result

def batch_triangulate(keypoints_, Pall, keypoints_pre=None, lamb=1e3):#输入(4,25,3)和相机参数(4,3,4)
    # keypoints: (nViews, nJoints, 3)
    # Pall: (nViews, 3, 4)
    # A: (nJoints, nViewsx2, 4), x: (nJoints, 4, 1); b: (nJoints, nViewsx2, 1)
    v = (keypoints_[:, :, -1]>0).sum(axis=0)#4个视角每个点的置信度，合成25个点的置信度
    valid_joint = np.where(v > 1)[0]
    keypoints = keypoints_[:, valid_joint]
    conf3d = keypoints[:, :, -1].sum(axis=0)/v[valid_joint]
    # P2: P矩阵的最后一行：(1, nViews, 1, 4)
    P0 = Pall[None, :, 0, :]#None是增加一个维度的意思
    P1 = Pall[None, :, 1, :]
    P2 = Pall[None, :, 2, :]
    # uP2: x坐标乘上P2: (nJoints, nViews, 1, 4)
    uP2 = keypoints[:, :, 0].T[:, :, None] * P2
    vP2 = keypoints[:, :, 1].T[:, :, None] * P2
    conf = keypoints[:, :, 2].T[:, :, None]
    Au = conf * (uP2 - P0)
    Av = conf * (vP2 - P1)
    A = np.hstack([Au, Av])
    if keypoints_pre is not None:
        # keypoints_pre: (nJoints, 4)
        B = np.eye(4)[None, :, :].repeat(A.shape[0], axis=0)
        B[:, :3, 3] = -keypoints_pre[valid_joint, :3]
        confpre = lamb * keypoints_pre[valid_joint, 3]
        # 1, 0, 0, -x0
        # 0, 1, 0, -y0
        # 0, 0, 1, -z0
        # 0, 0, 0,   0
        B[:, 3, 3] = 0
        B = B * confpre[:, None, None]
        A = np.hstack((A, B))
    u, s, v = np.linalg.svd(A)#奇异值分解
    X = v[:, -1, :]
    X = X / X[:, 3:]
    # out: (nJoints, 4)
    result = np.zeros((keypoints_.shape[1], 4))
    result[valid_joint, :3] = X[:, :3]
    result[valid_joint, 3] = conf3d
    return result

eps = 0.01
def simple_recon_person(keypoints_use, Puse):
    out = batch_triangulate(keypoints_use, Puse)#输入4个视角的关节点，相机参数，输出3D关节点25*4
    # compute reprojection error
    kpts_repro = projectN3(out, Puse)#利用3d关节点计算4个视角的重投影(4,25,3)
    square_diff = (keypoints_use[:, :, :2] - kpts_repro[:, :, :2])**2 #均方差
    conf = np.repeat(out[None, :, -1:], len(Puse), 0)#out[None, :, -1:]是3d点的置信度，在第0维复制4份(1, 25, 1)==》(4,25,1)
    kpts_repro = np.concatenate((kpts_repro, conf), axis=2)
    #https://blog.csdn.net/weixin_49804978/article/details/121922128，
    return out, kpts_repro#输出3d关节点(25,3+1)1代表置信度，4个视角的重投影(4,25,2+1+1)2代表已经除以深度后的像素坐标，前一个1代表深度s,后一个1代表置信度

def check_limb(keypoints3d, limb_means, thres=0.5):
    # keypoints3d: (nJ, 4)
    valid = True
    cnt = 0
    for (src, dst), val in limb_means.items():
        if not (keypoints3d[src, 3] > 0 and keypoints3d[dst, 3] > 0):
            continue
        cnt += 1 
        # 计算骨长
        l_est = np.linalg.norm(keypoints3d[src, :3] - keypoints3d[dst, :3])
        if abs(l_est - val['mean'])/val['mean']/val['std'] > thres:
            valid = False
            break
    # 至少两段骨头可以使用
    valid = valid and cnt > 2
    return valid
