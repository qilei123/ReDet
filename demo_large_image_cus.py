import cv2
import math
import mmcv
import numpy as np
import os
import pdb
from mmcv import Config
from tqdm import tqdm

import DOTA_devkit.polyiou as polyiou
from mmdet.apis import init_detector, inference_detector, draw_poly_detections
from mmdet.datasets import get_dataset
import glob

import datetime
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from Tracks import *
from pycocotools.coco import COCO

dota15_colormap = [
    (54, 67, 244),
    (99, 30, 233),
    (176, 39, 156),
    (183, 58, 103),
    (181, 81, 63),
    (243, 150, 33),
    (212, 188, 0),
    (136, 150, 0),
    (80, 175, 76),
    (74, 195, 139),
    (57, 220, 205),
    (59, 235, 255),
    (0, 152, 255),
    (34, 87, 255),
    (72, 85, 121),
    (139, 125, 96)]

classnames_ = ('Small 1-piece vehicle', 'Large 1-piece vehicle', 'Extra-large 2-piece truck', 'Tractor', 'Trailer', 'Motorcycle','mix')
classnames_ = ('Small 1-piece vehicle', 'Large 1-piece vehicle', 'Extra-large 2-piece truck')

def py_cpu_nms_poly_fast_np(dets, thresh):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                           dets[i][2], dets[i][3],
                                           dets[i][4], dets[i][5],
                                           dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou

        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


class DetectorModel():
    def __init__(self,
                 config_file,
                 checkpoint_file):
        # init RoITransformer
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.cfg = Config.fromfile(self.config_file)
        self.data_test = self.cfg.data['test']
        self.dataset = get_dataset(self.data_test)
        self.classnames = self.dataset.CLASSES
        print(self.classnames)
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')
        self.pre_detections = None
    def inference_single(self, imagname, slide_size, chip_size):
        img = mmcv.imread(imagname)
        height, width, channel = img.shape
        slide_h, slide_w = slide_size
        hn, wn = chip_size
        # TODO: check the corner case
        # import pdb; pdb.set_trace()
        total_detections = [np.zeros((0, 9)) for _ in range(len(self.classnames))]

        for i in range(int(width / slide_w )):
            for j in range(int(height / slide_h) ):
                subimg = np.zeros((hn, wn, channel))
                # print('i: ', i, 'j: ', j)
                chip = img[j * slide_h:j * slide_h + hn, i * slide_w:i * slide_w + wn, :3]
                subimg[:chip.shape[0], :chip.shape[1], :] = chip
                
                time0 = datetime.datetime.now()
                #print(subimg)
                chip_detections = inference_detector(self.model, subimg)
                #print(chip_detections)
                time1 = datetime.datetime.now()
                #print("--------inference_detector process--------")
                #print((time1-time0).microseconds/1000)
                
                # print('result: ', result)
                for cls_id, name in enumerate(self.classnames):
                    chip_detections[cls_id][:, :8][:, ::2] = chip_detections[cls_id][:, :8][:, ::2] + i * slide_w
                    chip_detections[cls_id][:, :8][:, 1::2] = chip_detections[cls_id][:, :8][:, 1::2] + j * slide_h
                    # import pdb;pdb.set_trace()
                    try:
                        total_detections[cls_id] = np.concatenate((total_detections[cls_id], chip_detections[cls_id]))
                    except:
                        import pdb;
                        pdb.set_trace()
        # nms
        #print(total_detections)
        for i in range(len(self.classnames)):
            keep = py_cpu_nms_poly_fast_np(total_detections[i], 0.1)

            total_detections[i] = total_detections[i][keep]
            #if i in [3,4]:
            #    total_detections[i] = []
        
        #total_detections[6] = np.concatenate((total_detections[0], total_detections[1],total_detections[2]))
        #keep = py_cpu_nms_poly_fast_np(total_detections[6], 0.01)
        #total_detections[6] = total_detections[6][keep]
        #print(total_detections[6])
        #print(total_detections)
        return total_detections

    def inference_single_vis(self, srcpath, dstpath, frame_size, slide_size, chip_size, speed=False):
        srcpath = mmcv.imread(srcpath)
        #srcpath = mmcv.imrotate(srcpath, -90,auto_bound=True)
        srcpath = cv2.resize(srcpath,(frame_size[1],frame_size[0]))
        
        time0 = datetime.datetime.now()
        detections = self.inference_single(srcpath, slide_size, chip_size)
        time1 = datetime.datetime.now()
        #print("--------inference_single process--------")
        #print((time1-time0).microseconds/1000)
        #img = srcpath
        img = draw_poly_detections(srcpath, detections, self.classnames, scale=1, threshold=0.2,
                                   colormap=dota15_colormap)
        if not dstpath==None:
            cv2.imwrite(dstpath, img)

        if speed:
            speeds = self.easySpeed(detections)
        return img,detections
    def easySpeed(self,detections):
        #print(detections)
        if self.pre_detections is not None:
            pass
        
        self.pre_detections = detections
        return None

def filt_detections(detections, class_names, scale=1, threshold=0.2,cat_ids = [6]):
    polygons = []
    for j, name in enumerate(classnames_):
        if j in cat_ids:
            dets = detections[j]
            
            for det in dets:
                bbox = det[:8] * scale
                score = det[-1]
                if score < threshold:
                    continue
                
                bbox = list(map(int, bbox))        
                bbox.append(score)
                bbox.append(j)
                polygons.append(bbox)
    #print(polygons)
    return polygons
def image_process():
    print("start initial detector")
    model_name = "ReDet_re50_refpn_1x_TD_ms"
    model = DetectorModel(
        r"configs/ReDet_trans_drone/"+model_name+".py",
        r"/data2/qilei_chen/DATA/trans_drone/work_dirs/"+model_name+"/latest.pth")
    print("end initial detector")

    img_dir = "/data2/qilei_chen/DATA/trans_drone/images"
    out_dir = "/data2/qilei_chen/DATA/trans_drone/work_dirs/"+model_name+"/results"
    #img_dir = "/data2/qilei_chen/DATA/usf_drone/images"
    #out_dir = "/data2/qilei_chen/DATA/trans_drone/work_dirs/"+model_name+"/usf_rotate_results"
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img_names = os.listdir(img_dir)
    for img_name in tqdm(img_names):
        if 'jpg' in img_name:
            img_path = os.path.join(img_dir, img_name)
            out_path = os.path.join(out_dir, img_name)
            model.inference_single_vis(img_path, out_path, (540, 960),(540, 960), (540, 960))

def single_video_process(model,cap1,cap2,frame_size=(540,960),dst_dir=None):
    
    tmer = tracks_manager()

    success,frame = cap1.read()
    frame_index = 1
    while success:
        result_frame,detections = model.inference_single_vis(frame,"/data2/qilei_chen/DATA/trans_drone/videos/results2/test1.jpg",frame_size,frame_size,frame_size)
        #print(detections)
        det_polygons = filt_detections(detections,model.classnames,cat_ids=[1,2,3]) 
        
        tmer.update(det_polygons,frame_index)

        result_frame = tmer.vis(result_frame)

        cv2.imwrite("/data2/qilei_chen/DATA/trans_drone/videos/results2/test.jpg",result_frame)
        cap2.write(result_frame)
        #print(frame_index)
        frame_index+=1
        #if frame_index==20:
        #    break
        success,frame = cap1.read()
    if isinstance( dst_dir,str):
        tmer.save_results(dst_dir+".pkl")
def videos_process(src_dir,dst_dir):
    print("start initial detector")
    #model_name = "ReDet_re50_refpn_1x_TD_ms_3cat"
    model_name = "retinanet_obb_r50_fpn_2x_TD_3cat_scratch"
    model = DetectorModel(
        r"configs/ReDet_trans_drone/"+model_name+".py",
        r"/data2/qilei_chen/DATA/trans_drone/work_dirs/retinanet_obb_r50_fpn_2x_TD_3cat_stratch/latest.pth")
    print("end initial detector") 

    video_dir_list = glob.glob(os.path.join(src_dir,"*.MOV")) 
    for src_dir in video_dir_list:
        print(src_dir)
        src_cap = cv2.VideoCapture(src_dir)
        
        fps = src_cap.get(cv2.CAP_PROP_FPS)
        frame_size = (540,960)
        dst_dir_name = os.path.join(dst_dir,os.path.basename(src_dir).replace("MOV","avi"))
        model.pre_detections = None
        #if not os.path.exists(dst_dir_name):
        if True:
            dst_writer = cv2.VideoWriter(dst_dir_name, cv2.VideoWriter_fourcc("P", "I", "M", "1"), fps, (frame_size[1],frame_size[0]))
            single_video_process(model,src_cap,dst_writer,frame_size,dst_dir_name)
        #break

def show_pickle():
    dst_dir = "/data2/qilei_chen/DATA/trans_drone/videos/results1/DJI_0003 400 90 degree.avi.pkl"
    with open(dst_dir, 'rb') as handle:
        pk = pickle.load(handle)
        print(pk)

def show_gts():
    anns_file = "/data2/qilei_chen/DATA/trans_drone/annotations/merged_annotations_val3.json"
    coco_instance = COCO(anns_file)
    coco_imgs = coco_instance.imgs
    for key in coco_imgs:
        annIds = coco_instance.getAnnIds(imgIds=coco_imgs[key]['id'])
        anns = coco_instance.loadAnns(annIds)  
        img_file_name = coco_imgs[key]["file_name"]
        img_dir = os.path.join(
            "/data2/qilei_chen/DATA/trans_drone/images", img_file_name)
        img = mmcv.imread(img_dir)
        for ann in anns:
            [x, y, w, h] = ann['bbox']
            cv2.rectangle(img, (int(x), int(y)),
                        (int(x+w), int(y+h)), (0, 255, 0), 2)   
            bbox = list(map(int, ann['segmentation'][0])) 
            if len(bbox)==8:
                for i in range(3):
                    cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i+1) * 2], bbox[(i+1) * 2 + 1]), color=(0, 255, 0), thickness=2,lineType=cv2.LINE_AA)
                cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=(0, 255, 0), thickness=2,lineType=cv2.LINE_AA) 
        cv2.imwrite("/data2/qilei_chen/DATA/trans_drone/videos/results2/gts/"+img_file_name,img)       
if __name__ == '__main__':
    #image_process()
    videos_process("/data2/qilei_chen/DATA/trans_drone/videos/rounds","/data2/qilei_chen/DATA/trans_drone/videos/results2")
    #show_pickle()
    #show_gts()
