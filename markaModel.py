import random
import cv2
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
class_dicto = {0:"seat_belt_on",1:"seat_belt_off"}
# import some common libraries
import numpy as np
import os, json, cv2, random
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from matplotlib import pyplot as plt
from     detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import json
import base64
import argparse
import glob
def im2json(im):
    _, imdata = cv2.imencode('.JPG',im)
    jstr = base64.b64encode(imdata).decode('ascii')
    return jstr

def getParser():
    args = argparse.ArgumentParser()
    args.add_argument("--im_path",type=str)
    args.add_argument("--out_path",type=str,default="./labels")
    return args.parse_args()
for d in ["train", "val"]:
    DatasetCatalog.register("seatbelt_" + d, lambda d=d: get_seat_belt_dicts(d))
    MetadataCatalog.get("seatbelt_" + d).set(thing_classes=["seat_belt_on", "seat_belt_off"])
balloon_metadata = MetadataCatalog.get("seatbelt_val")
args = getParser()
to_read = os.path.join(args.im_path,"*")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("seatbelt_train",)
cfg.DATASETS.TEST = ("seatbelt_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 14
cfg.SOLVER.BASE_LR = 1e-4  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon)
cfg.TEST.EVAL_PERIOD = 100
cfg.MODEL.WEIGHTS = os.path.join("./output", "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set a custom testing threshold
predictor = DefaultPredictor(cfg)

cfgz = get_cfg()
cfgz.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfgz.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfgz.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
car_predictor = DefaultPredictor(cfgz)
if not os.path.isdir(args.out_path):
    os.makedirs(args.out_path)
for im_f in glob.glob(to_read):
    try:


        im = cv2.imread(im_f)
        label_me_dicto = {}
        im_data = im2json(im)
        label_me_dicto["version"] = "3.16.7"
        label_me_dicto["flags"] = {}
        label_me_dicto["imageData"] = im_data

        shapes = []
        label_me_dicto["lineColor"] = [0,255,0,128]
        label_me_dicto["fillColor"] = [255,0,0,128]
        label_me_dicto["imagePath"] = im_f
        label_me_dicto["imageHeight"] = im.shape[0]
        label_me_dicto["imageWidth"] = im.shape[1]







        outputs = car_predictor(im)
        bbox = outputs["instances"].to("cpu").pred_boxes.tensor.numpy()
        classes = outputs["instances"].to("cpu").pred_classes.numpy()

        candidates = []

        if 2 in classes:
            candidates += list(bbox[np.where(classes == 2)])
        if 7 in classes:
            candidates += list(bbox[np.where(classes == 7)])
        if 5 in classes:
            candidates += list(bbox[np.where(classes == 5)])

        candidates = np.array(candidates)
        print(candidates
              )
        for box in candidates:
            car_dicto = {}
            car_dicto["label"] = "car"
            car_dicto["line_color"] = None
            car_dicto["fill_color"] = None
            car_dicto["points"] = [[float(box[0]),float(box[1])],[float(box[2]),float(box[3])]]
            car_dicto["shape_type"] = "rectangle"
            car_dicto["flags"] = {}
            shapes.append(car_dicto)
            car_box = list(map(int,box))

            car_im = im[car_box[1]:car_box[3]+1,car_box[0]:car_box[2]+1]
            kemer_outputs = predictor(car_im)
            v = Visualizer(car_im[:, :, ::-1],
                           metadata=balloon_metadata,
                           scale=0.5,
                            # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
            print(kemer_outputs["instances"].to("cpu"))
            out = v.draw_instance_predictions(kemer_outputs["instances"].to("cpu"))
            plt.imshow(out.get_image()[:, :, ::-1])
            plt.pause(0.001)

            kemer_bbox = kemer_outputs["instances"].to("cpu").pred_boxes.tensor.numpy()
            kemer_classes = kemer_outputs["instances"].to("cpu").pred_classes.numpy()
            for q,kemer_box in enumerate(kemer_bbox):
                kemer_box[0] += box[0]
                kemer_box[1] += box[1]
                disp_x2 = abs(box[2] - car_im.shape[1])
                disp_y2 = abs(box[3]-  car_im.shape[0])
                kemer_box[2] += disp_x2
                kemer_box[3] += disp_y2

                kemer_box_draw = list(map(int,kemer_box))

                kemer_dicto = {}
                kemer_dicto["label"] = "person"
                kemer_dicto["line_color"] = None
                kemer_dicto["fill_color"] = None
                kemer_box = list(map(str,kemer_box))
                kemer_dicto["points"] = [[float(kemer_box[0]), float(kemer_box[1]), float(kemer_box[2]), float(kemer_box[3])]]
                kemer_dicto["shape_type"] = "rectangle"
                kemer_dicto["flags"] = {"seatbelt":1 if kemer_classes[q] == 0 else -1}
                shapes.append(kemer_dicto)
        label_me_dicto["shapes"] = shapes
        json.dump(label_me_dicto,open(os.path.join(args.out_path,im_f.split("/")[-1]) + ".json","