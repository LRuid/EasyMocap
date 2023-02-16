'''
  @ Date: 2021-01-13 20:38:33
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-04-13 21:43:52
  @ FilePath: /EasyMocapRelease/scripts/preprocess/extract_video.py
'''
import os, sys
import cv2
from os.path import join
from tqdm import tqdm
from glob import glob
import numpy as np
from multiprocessing import Process,current_process

mkdir = lambda x: os.makedirs(x, exist_ok=True)#匿名函数 返回值mkdir是一个函数，参数是x 等价于 mkdir(x) = os.makedirs(x, exist_ok=True)

def extract_video(videoname, path, start, end, step):
    base = os.path.basename(videoname).replace('.mp4', '')
    if not os.path.exists(videoname):#判断视频是否存在，不存在直接返回base
        return base
    outpath = join(path, 'images', base)
    if os.path.exists(outpath) and len(os.listdir(outpath)) > 0:#判断图片文件夹是否被创建出来
        num_images = len(os.listdir(outpath))#判断图片文件夹n中的图片个数
        print('>> exists {} frames'.format(num_images))#已经存在多少帧
        return base
    else:
        os.makedirs(outpath, exist_ok=True)#创建文件夹
    video = cv2.VideoCapture(videoname)#opencv传入视频
    totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))#获得视频帧数
    for cnt in tqdm(range(totalFrames), desc='{:10s}'.format(os.path.basename(videoname)),position=int(base)):#desc（'str'）: 传入进度条的前缀
        ret, frame = video.read()#ret 取得帧正确返回true
        if cnt < start:continue
        if cnt >= end:break
        if not ret:continue
        cv2.imwrite(join(outpath, '{:06d}.jpg'.format(cnt)), frame)
    video.release()
    return base

def extract_2d(openpose, image, keypoints, render, args):
    skip = False
    if os.path.exists(keypoints):#判断/home/lrd/data/lrd/easymocap_filer/zju_data/openpose/1是否存在
        # check the number of images and keypoints
        if len(os.listdir(image)) == len(os.listdir(keypoints)):#判断image/1和openpose/1中文件个数是否相等
            skip = True#相等就略过
    if not skip:#没有略过
        os.makedirs(keypoints, exist_ok=True)#创建/home/lrd/data/lrd/easymocap_filer/zju_data/openpose/1文件夹
        if os.name != 'nt':#如果不是windows系统则运行
            cmd = './build/examples/openpose/openpose.bin --image_dir {} --write_json {} --display 0'.format(image, keypoints)
        else:
            cmd = 'bin\\OpenPoseDemo.exe --image_dir {} --write_json {} --display 0'.format(join(os.getcwd(),image), join(os.getcwd(),keypoints))
        if args.highres!=1:
            cmd = cmd + ' --net_resolution -1x{}'.format(int(16*((368*args.highres)//16)))
        if args.handface:
            cmd = cmd + ' --hand --face'
        if args.render:
            if os.path.exists(join(os.getcwd(),render)):
                cmd = cmd + ' --write_images {}'.format(join(os.getcwd(),render))
            else:
                os.makedirs(join(os.getcwd(),render), exist_ok=True)
                cmd = cmd + ' --write_images {}'.format(join(os.getcwd(),render))
        else:
            cmd = cmd + ' --render_pose 0'
        os.chdir(openpose)#回到/home/lrd/data/lrd/easymocap_filer/openpose目录下
        os.system(cmd)
#运行'./build/examples/openpose/openpose.bin --image_dir /home/lrd/data/lrd/easymocap_filer/zju_data/images/1 --write_json /home/lrd/data/lrd/easymocap_filer/zju_data/openpose/1 --display 0 --render_pose 0'

import json
def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(file, data):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def create_annot_file(annotname, imgname):
    assert os.path.exists(imgname), imgname
    img = cv2.imread(imgname)
    height, width = img.shape[0], img.shape[1]
    imgnamesep = imgname.split(os.sep)#分割文件名['', 'home', 'lrd', 'data', 'lrd', 'easymocap_filer', 'zju_data', 'images', '1', '000000.jpg']
    filename = os.sep.join(imgnamesep[imgnamesep.index('images'):])#os.sep.join(imgnamesep[7:])
    annot = {
        'filename':filename,
        'height':height,
        'width':width,
        'annots': [],
        'isKeyframe': False
    }
    save_json(annotname, annot)
    return annot

def bbox_from_openpose(keypoints, rescale=1.2, detection_thresh=0.01):
    """Get center and scale for bounding box from openpose detections."""
    valid = keypoints[:,-1] > detection_thresh#[0.897879, 0.93068, 0.804994, ...] > 0.01 ?
    valid_keypoints = keypoints[valid][:,:-1]#(24,2)去除了置信度以及一个置信度为0的点
    center = valid_keypoints.mean(axis=0)#压缩行，对列求平均array([478.22770833, 488.11816667])
    bbox_size = valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)#识别行，对列求最大、最小值，求差
    # adjust bounding box tightness
    bbox_size = bbox_size * rescale#box的大小 高*宽
    bbox = [
        center[0] - bbox_size[0]/2, #中心像素的高-盒子高的一半
        center[1] - bbox_size[1]/2,
        center[0] + bbox_size[0]/2, 
        center[1] + bbox_size[1]/2,
        keypoints[valid, 2].mean()
    ]
    return bbox

def load_openpose(opname):
    mapname = {'face_keypoints_2d':'face2d', 'hand_left_keypoints_2d':'handl2d', 'hand_right_keypoints_2d':'handr2d'}
    assert os.path.exists(opname), opname#'/home/lrd/data/lrd/easymocap_filer/zju_data/openpose/1/000000_keypoints.json'判断是否存在
    data = read_json(opname)
    out = []
    pid = 0
    for i, d in enumerate(data['people']):#
        keypoints = d['pose_keypoints_2d']
        keypoints = np.array(keypoints).reshape(-1, 3)#(25*3)
        annot = {
            'bbox': bbox_from_openpose(keypoints),#算出关节点的box
            'personID': pid + i,
            'keypoints': keypoints.tolist(),#将np对象转换成list
            'isKeyframe': False
        }
        for key in ['face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']:#没有这些东西
            if len(d[key]) == 0:
                continue
            kpts = np.array(d[key]).reshape(-1, 3)
            annot[mapname[key]] = kpts.tolist()
        out.append(annot)
    return out
    
def convert_from_openpose(path_orig, src, dst, annotdir):
    # convert the 2d pose from openpose
    os.chdir(path_orig)#进入'/data/lrd/easymocap_filer'
    inputlist = sorted(os.listdir(src))#排序'/home/lrd/data/lrd/easymocap_filer/zju_data/openpose/1' ，800个json
    for inp in tqdm(inputlist, desc='{:10s}'.format(os.path.basename(dst))):#标注过程
        annots = load_openpose(join(src, inp))#关节点生成的box标注信息
        base = inp.replace('_keypoints.json', '')#
        annotname = join(dst, base+'.json')#'/home/lrd/data/lrd/easymocap_filer/zju_data/annots/1/000000.json'
        imgname = annotname.replace(annotdir, 'images').replace('.json', '.jpg')#'/home/lrd/data/lrd/easymocap_filer/zju_data/images/1/000000.jpg'
        annot = create_annot_file(annotname, imgname)
        annot['annots'] = annots
        save_json(annotname, annot)

def detect_frame(detector, img, pid=0):
    lDetections = detector.detect([img])[0]
    annots = []
    for i in range(len(lDetections)):
        annot = {
            'bbox': [float(d) for d in lDetections[i]['bbox']],
            'personID': pid + i,
            'keypoints': lDetections[i]['keypoints'].tolist(),
            'isKeyframe': True
        }
        annots.append(annot)
    return annots

config_high = {
    'yolov4': {
            'ckpt_path': 'data/models/yolov4.weights',
            'conf_thres': 0.3,
            'box_nms_thres': 0.5 # 阈值=0.9，表示IOU 0.9的不会被筛掉
    },
    'hrnet':{
        'nof_joints': 17,
        'c': 48,
        'checkpoint_path': 'data/models/pose_hrnet_w48_384x288.pth'
    },
    'detect':{
        'MIN_PERSON_JOINTS': 10,
        'MIN_BBOX_AREA': 5000,
        'MIN_JOINTS_CONF': 0.3,
        'MIN_BBOX_LEN': 150
    }
}

config_low = {
    'yolov4': {
        'ckpt_path': 'data/models/yolov4.weights',
        'conf_thres': 0.1,
        'box_nms_thres': 0.9 # 阈值=0.9，表示IOU 0.9的不会被筛掉
    },
    'hrnet':{
        'nof_joints': 17,
        'c': 48,
        'checkpoint_path': 'data/models/pose_hrnet_w48_384x288.pth'
    },
    'detect':{
        'MIN_PERSON_JOINTS': 0,
        'MIN_BBOX_AREA': 0,
        'MIN_JOINTS_CONF': 0.0,
        'MIN_BBOX_LEN': 0
    }
}

def extract_yolo_hrnet(image_root, annot_root, ext='jpg', use_low=False):
    imgnames = sorted(glob(join(image_root, '*.{}'.format(ext))))
    import torch
    device = torch.device('cuda')
    from easymocap.estimator import Detector
    config = config_low if use_low else config_high
    print(config)
    detector = Detector('yolo', 'hrnet', device, config)
    for nf, imgname in enumerate(tqdm(imgnames)):
        annotname = join(annot_root, os.path.basename(imgname).replace('.{}'.format(ext), '.json'))
        annot = create_annot_file(annotname, imgname)
        img0 = cv2.imread(imgname)
        annot['annots'] = detect_frame(detector, img0, 0)
        for i in range(len(annot['annots'])):
            x = annot['annots'][i]
            x['area'] = max(x['bbox'][2] - x['bbox'][0], x['bbox'][3] - x['bbox'][1])**2
        annot['annots'].sort(key=lambda x:-x['area'])
        # 重新赋值人的ID
        for i in range(len(annot['annots'])):
            annot['annots'][i]['personID'] = i
        save_json(annotname, annot)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help="the path of data",default='/home/lrd/data/lrd/easymocap_filer/zju_data')
    parser.add_argument('--mode', type=str, default='openpose', choices=['openpose', 'yolo-hrnet'], help="model to extract joints from image")
    parser.add_argument('--ext', type=str, default='jpg', choices=['jpg', 'png'], help="image file extension")
    parser.add_argument('--annot', type=str, default='annots', help="sub directory name to store the generated annotation files, default to be annots")
    parser.add_argument('--highres', type=float, default=1)
    parser.add_argument('--handface', action='store_true')#表示命令行只要输入了--handface ，handface=True
    parser.add_argument('--openpose', type=str, 
        default='/home/lrd/data/lrd/easymocap_filer/openpose')
    parser.add_argument('--render', action='store_true', 
        help='use to render the openpose 2d')
    parser.add_argument('--no2d', action='store_true',
        help='only extract the images')
    parser.add_argument('--start', type=int, default=0,
        help='frame start')
    parser.add_argument('--end', type=int, default=10000,
        help='frame end')    
    parser.add_argument('--step', type=int, default=1,
        help='frame step')
    parser.add_argument('--low', action='store_true',
        help='decrease the threshold of human detector')
    parser.add_argument('--gtbbox', action='store_true',
        help='use the ground-truth bounding box, and hrnet to estimate human pose')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--path_origin', default=os.getcwd())#返回当前工作目录
    args = parser.parse_args()
    mode = args.mode

    if os.path.isdir(args.path):#判断path是不是文件夹，如果是
        image_path = join(args.path, 'images')#连接文件夹和images
        os.makedirs(image_path, exist_ok=True)#创建images文件夹，如果exist_ok为True，则在目标目录已存在的情况下不会触发FileExistsError异常。
        subs_image = sorted(os.listdir(image_path))#图片地址排序
        subs_videos = sorted(glob(join(args.path, 'videos', '*.mp4')))#用法：glob( path + '*.某格式' ),返回list videos下的所有MP4文件
        if len(subs_videos) > len(subs_image):#判断视频和图片文件夹的个数是否相等
            videos = sorted(glob(join(args.path, 'videos', '*.mp4')))#为什么不直接用subs_videos？
            subs = []
            cameras_extract_nums = []
            for video in videos:
                p = Process(target=extract_video,args=(video, args.path, args.start, args.end, args.step))
                #basename = extract_video(video, args.path, start=args.start, end=args.end, step=args.step)
                cameras_extract_nums.append(p)
                subs.append(p.name.replace('Process-',''))
                p.start()
            for process_example in cameras_extract_nums:
                process_example.join()
        else:
            subs = sorted(os.listdir(image_path))
        print('cameras: ', ' '.join(subs))#10.24完成多进程提取视频
        if not args.no2d:#如果没有开启 no2d 则运行。就是要检测2d
            for sub in subs:
                image_root = join(args.path, 'images', sub)#图片1文件夹'/home/lrd/data/lrd/easymocap_filer/zju_data/images/1'
                annot_root = join(args.path, args.annot, sub)#标注1文件夹'/home/lrd/data/lrd/easymocap_filer/zju_data/annots/1'
                if os.path.exists(annot_root):#判断标注文件夹是否存在
                    # check the number of annots and images
                    if len(os.listdir(image_root)) == len(os.listdir(annot_root)):#图片1中文件数和标注1中文件数是否相等
                        print('skip ', annot_root)#相等就跳过
                        continue
                if mode == 'openpose':#判断是否用的openpose
                    #提取2d数据到openpose文件夹中
                    extract_2d(args.openpose, image_root, 
                        join(args.path, 'openpose', sub), 
                        join(args.path, 'openpose_render', sub), args)
                    #将2d数据转换成标注数据
                    convert_from_openpose(
                        path_orig=args.path_origin,
                        src=join(args.path, 'openpose', sub),
                        dst=annot_root,
                        annotdir=args.annot
                    )
                elif mode == 'yolo-hrnet':
                    extract_yolo_hrnet(image_root, annot_root, args.ext, args.low)
    else:
        print(args.path, ' not exists')
