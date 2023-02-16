'''
  @ Date: 2021-08-19 22:06:22
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-05-23 22:58:57
  @ FilePath: /EasyMocapPublic/apps/preprocess/extract_keypoints.py
'''
import os,sys,time
from os.path import join
from tkinter import image_names
from tqdm import tqdm
import numpy as np
from multiprocessing import Process,JoinableQueue,Value
from easymocap.mytools import simple_recon_person,read_camera
from easymocap.socket.base_client import BaseSocketClient
import cv2,json
def load_subs(path, subs):
    if len(subs) == 0:
        subs = sorted(os.listdir(join(path, 'images')))#排序
    # subs = [sub for sub in subs if os.path.isdir(join(path, 'images', sub))]
    # if len(subs) == 0:
    #     subs = ['']
    return subs

def read_json(path):#传入文件输出字典
    assert os.path.exists(path), path
    with open(path) as f:
        try:
            data = json.load(f)
        except:
            print('Reading error {}'.format(path))
            data = []
    return data

def read_cameras(path,cams):
    # 读入相机参数
    intri_name = join(path, 'intri.yml')
    extri_name = join(path, 'extri.yml')
    if os.path.exists(intri_name) and os.path.exists(extri_name):
        cameras = read_camera(intri_name, extri_name)
        cameras.pop('basenames')
        # 注意：这里的相机参数一定要用定义的，不然只用一部分相机的时候会出错
        cams_nums = cams
        Pall = np.stack([cameras[cam]['P'] for cam in cams_nums])
        return Pall
    else:
        print('\n!!!\n!!!there is no camera parameters, maybe bug: \n', intri_name, extri_name, '\n')
        cameras = None

def check_repro_error(keypoints3d, kpts_repro, keypoints2d,projection_matrix,MAX_REPRO_ERROR=50):
    """
    检查重投影误差
    input:3d data,重投影,2d data,
    """
    square_diff = (keypoints2d[:, :, :2] - kpts_repro[:, :, :2])**2 
    conf = keypoints3d[None, :, -1:]#(1,25,1)取出3d点的置信度
    conf = (keypoints3d[None, :, -1:] > 0) * (keypoints2d[:, :, -1:] > 0)#(4,25,1)1=true or false 代表该点是否存在
    #4个视角25个点的误差
    dist = np.sqrt((((kpts_repro[..., :2] - keypoints2d[..., :2])*conf)**2).sum(axis=-1))#sqrt((x1-x2)^2+(y1-y2)^2)
    vv, jj = np.where(dist > MAX_REPRO_ERROR)
    if vv.shape[0] > 0:#如果误差过大就重新重构
        keypoints2d[vv, jj, -1] = 0.
        keypoints3d, kpts_repro = simple_recon_person(keypoints2d, projection_matrix)
    return keypoints3d, kpts_repro

sys.path.append('/usr/local/python')
from openpose import pyopenpose as op
def run_op_from_images(image_root,model_folder,q):
    """
    单张图运行openpose Ruid 2022.11.12
    input:输入图片文件夹 /.../easymocap_filer/zju_data/images/1
    output:依次序检测出关节点  <class 'numpy.ndarray'> 
    """
    opWrapper = op.WrapperPython()
    opWrapper.configure(model_folder)
    opWrapper.start()
    imagePaths = op.get_images_on_directory(image_root)
    for imagePath in imagePaths:
        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        q.put(datum.poseKeypoints)
        print("Body keypoints: \n" + str(datum.poseKeypoints))
    q.join()

def run_op_from_image(image_root,model_folder,q):
    #debug使用 检测单张图
    opWrapper = op.WrapperPython()
    opWrapper.configure(model_folder)
    opWrapper.start()
    datum = op.Datum()
    imageToProcess = cv2.imread(image_root)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    q.put(datum.poseKeypoints)
    print("Body keypoints: \n" + str(datum.poseKeypoints))

def read_kpt2d_from_folder(kpt2d_folder,q):
    imgnames = sorted(os.listdir(kpt2d_folder))
    for kpt2d in imgnames:
        annot_kpt2d = read_json(join(kpt2d_folder,kpt2d))
        people_1 = annot_kpt2d["people"][0]#有时候会检测出多个人，我这里只选用第一个人
        people_1_kpt2d = np.array(people_1['pose_keypoints_2d']).reshape(1,25,3)
        q.put(people_1_kpt2d)



def send_keypoints_form_queues(cout,projection_matrix,queue_nums,check_repro=True):
    """
    接收多个队列和多视图投影矩阵,分别从多个队列中取数据并且计算3d data
    """ 
    data = [dict()]#这里是为了与可视化的代码的输入数据对齐
    data[0]['id']=0
    client = BaseSocketClient('127.0.0.1', 9999)#开启客户端，可视化用
    while 1:
        keypoints = []
        for q in queue_nums:
            r = q.get()[0]#由于可能检测到多人，选择第一个人Ruid 11.16 debug
            keypoints.append(r)
            q.task_done()
        if keypoints is not None:
            keypoints = np.stack(keypoints)
            keypoints3d, kpts_repro = simple_recon_person(keypoints, projection_matrix)
            if check_repro:
                keypoints3d, kpts_repro = check_repro_error(keypoints3d, kpts_repro, keypoints,projection_matrix,50)
            data[0]['keypoints3d']=keypoints3d
            client.send(data)#发送数据到服务端，可视化用

            #debug用 这里出现了一个问题 第485张图openpoes检测出了两个人，已做 Ruid 11.16
            print(keypoints3d)
            cout.value += 1
            print(cout.value)
            if cout.value == 485:
                b = 1
                continue
        
        
#ruid 11.15 22.52 待做
config = {
    'openpose':{
        'root': '',
        'res': 1,
        'hand': False,
        'face': False,
        'vis': False,
        'ext': '.jpg'
    },
    'openposecrop': {},
    'feet':{
        'root': '',
        'res': 1,
        'hand': False,
        'face': False,
        'vis': False,
        'ext': '.jpg'
    },
    'feetcrop':{
        'root': '',
        'res': 1,
        'hand': False,
        'face': False,
        'vis': False,
        'ext': '.jpg'
    },
    'yolo':{
        'ckpt_path': 'data/models/yolov4.weights',
        'conf_thres': 0.3,
        'box_nms_thres': 0.5, # means keeping the bboxes that IOU<0.5
        'ext': '.jpg',
        'isWild': False,
    },
    'hrnet':{
        'nof_joints': 17,
        'c': 48,
        'checkpoint_path': 'data/models/pose_hrnet_w48_384x288.pth'
    },
    'yolo-hrnet':{},
    'mp-pose':{
        'model_complexity': 2,
        'min_detection_confidence':0.5,
        'min_tracking_confidence': 0.5
    },
    'mp-holistic':{
        'model_complexity': 2,
        # 'refine_face_landmarks': True,
        'min_detection_confidence':0.5,
        'min_tracking_confidence': 0.5
    },
    'mp-handl':{
        'model_complexity': 1,
        'min_detection_confidence':0.3,
        'min_tracking_confidence': 0.1,
        'static_image_mode': False,
    },
    'mp-handr':{
        'model_complexity': 1,
        'min_detection_confidence':0.3,
        'min_tracking_confidence': 0.1,
        'static_image_mode': False,
    },
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default=None, help="the path of data")
    parser.add_argument('--subs', type=str, nargs='+', default=[], help="the path of data")
    # Output Control
    parser.add_argument('--annot', type=str, default='annots', 
        help="sub directory name to store the generated annotation files, default to be annots")
    # Detection Control
    parser.add_argument('--mode', type=str, default='openpose', choices=[
        'openpose', 'feet', 'feetcrop', 'openposecrop',
        'yolo-hrnet', 'yolo', 'hrnet', 
        'mp-pose', 'mp-holistic', 'mp-handl', 'mp-handr', 'mp-face'], 
        help="model to extract joints from image")
    # Openpose
    parser.add_argument('--openpose', type=str, 
        default=os.environ.get('openpose', 'openpose'))
    parser.add_argument('--openpose_res', type=float, default=1)
    parser.add_argument('--gpus', type=int, default=[], nargs='+')
    parser.add_argument('--ext', type=str, default='.jpg')
    parser.add_argument('--tmp', type=str, default=os.path.abspath('tmp'))
    parser.add_argument('--hand', action='store_true')
    parser.add_argument('--face', action='store_true')
    parser.add_argument('--wild', action='store_true',
        help='remove crowd class of yolo')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--force', action='store_true')
    #优化器阈值
    parser.add_argument('--MAX_REPRO_ERROR', type=int,
        help='The threshold of reprojection error', default=50)
    args = parser.parse_args()
    config['yolo']['isWild'] = args.wild
    mode = args.mode
    #如果没有图片，会从视频提取图片
    # if not os.path.exists(join(args.path, 'images')) and os.path.exists(join(args.path, 'videos')):
    #     # default extract image
    #     cmd = f'''python EasyMocap/apps/preprocess/extract_image.py {args.path}'''
    #     os.system(cmd)
    #subs = load_subs(args.path, args.subs)
    subs = ['1','2','3','4']
    if len(args.gpus) != 0:#开多GPU从图片中提取2d点并保存json
        # perform multiprocess by runcmd
        from easymocap.mytools.debug_utils import run_cmd
        nproc = len(args.gpus)
        plist = []
        for i in range(nproc):
            if len(subs[i::nproc]) == 0:
                continue
            cmd = f'export CUDA_VISIBLE_DEVICES={args.gpus[i]} && python3 EasyMocap/apps/preprocess/extract_keypoints.py {args.path} --mode {args.mode} --subs {" ".join(subs[i::nproc])}'
            cmd += f' --annot {args.annot} --ext {args.ext}'
            if args.hand:
                cmd += ' --hand'
            if args.face:
                cmd += ' --face'            
            if args.force:
                cmd += ' --force'
            cmd += f' --tmp {args.tmp}'
            cmd += ' &'
            print(cmd)
            p = run_cmd(cmd, bg=False)
            plist.extend(p)
        for p in plist:
            p.join()
        exit()#退出程序
    if len(subs) == 0:
        subs = ['']


    cout = Value('i', 0)#打开一个值共享，测试q.task_done()的功能
    global_tasks = []
    queue_nums = []
    params = dict()
    params["model_folder"] = "/home/lrd/data/lrd/easymocap_filer/openpose/models"
    Pall = read_cameras(args.path,subs[:4])#需要几个相机就调整相机个数subs[:2]代表2个相机
    start = time.time()
    for sub in subs[:4]:#需要几个相机就调整相机个数subs[:2]代表2个相机
        # config[mode]['force'] = args.force
        image_root = join(args.path, 'images', sub)
        kpt2d_annot = join(args.path, 'openpose', sub)
        if mode == 'openpose':
            q = JoinableQueue()
            queue_nums.append(q)
            p = Process(target=read_kpt2d_from_folder,args=(kpt2d_annot,q))
            #p = Process(target=run_op_from_images,args=(image_root,params,q))
            p.start()
            global_tasks.append(p)
            # global_tasks = extract_2d(image_root, annot_root, tmp_root, config[mode])
    receive_keypoints = Process(target=send_keypoints_form_queues,args=(cout,Pall,queue_nums,True))   
    receive_keypoints.daemon = True
    receive_keypoints.start()
    for task in global_tasks:
        task.join()
    end = time.time()
    print(end-start)