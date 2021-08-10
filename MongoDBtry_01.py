#
# This is a sample Python script.
#
import io, base64
import sys, os, json
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pymongo import MongoClient
import gridfs
# from time import clock
from uuid import uuid4
import pickle
from PIL import Image
import cv2
from datetime import datetime

import numpy as np
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
from sshtunnel import SSHTunnelForwarder
import pymongo
import pprint
from ssh_pymongo import MongoSession


"""
Main steps for Big Data Hndling
    1. Generating json files
    2. keeping the on MongoDB
"""


class mongoHandler:
    """
     Creating mongo database
    """
    def __init__(self, dbname):
        MONGO_HOST = "10.6.1.104"
        MONGO_USER = "gpu2"
        MONGO_PASS = "aAzZ"
        server = SSHTunnelForwarder(
            MONGO_HOST,
            ssh_username=MONGO_USER,
            ssh_password=MONGO_PASS,
            remote_bind_address=('127.0.0.1', 27017)
        )
        server.start()
        client = MongoClient('127.0.0.1', server.local_bind_port)

        self.db = client[dbname]


    def geDB(self):
        return self.db


colorList = ['black', 'white', 'red', 'blue', 'gray', 'yellow', 'green', 'darkGray', 'darkRed', 'orange','pink']
markList = ['toyota','citroen', 'kia', 'hyundai', 'bmw','mercedes', 'audi','honda', 'mazda', 'peugeot', 'lexus', 'subaru','opel' ,'renault','infiniti','fiat', 'volkswagen', 'skoda', 'mitsubishi' ]

class data2json:
    def __init__(self):

        # creating template
        self.data_template = {
            "tempId": "1",
            "version": "1.0.0",
            "date": "",
            "dataName": "",
            "flags": {
                "data_type": "image",
                "datasetName": ""
            },

            "description": {
                "content": "vehicle",
                "format": "jpg",
                "imageHeight": 0,
                "imageWidth": 0,
                "imageChannels": 1,
                'sizeof': 0,
            },

            "annotation": {"flags": {"num_objects": 1,
                                     "group_id": None,  # generate group id and describe the here
                                     },
                           # attributes

                           "objects": []  # objects closed
                           },  # annotation closed

            "sentiments": None,
            "description": {}
        }




    def img_data_to_pil(self, img_data):
        f = io.BytesIO()
        f.write(img_data)
        img_pil = PIL.Image.open(f)
        return img_pil

    def img_data_to_arr(self, img_data):
        img_pil = self.img_data_to_pil(img_data)
        img_arr = np.array(img_pil)
        return img_arr

    def img_b64_to_arr(self, img_b64):
        img_data = base64.b64decode(img_b64)
        img_arr = self.img_data_to_arr(img_data)
        return img_arr

    def img_pil_to_data(self, img_pil):
        f = io.BytesIO()
        img_pil.save(f, format="PNG")
        img_data = f.getvalue()
        return img_data

    def masks_to_bboxes(masks):
        if masks.ndim != 3:
            raise ValueError(
                "masks.ndim must be 3, but it is {}".format(masks.ndim)
            )
        if masks.dtype != bool:
            raise ValueError(
                "masks.dtype must be bool type, but it is {}".format(masks.dtype)
            )
        bboxes = []
        for mask in masks:
            where = np.argwhere(mask)
            (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
            bboxes.append((y1, x1, y2, x2))
        bboxes = np.asarray(bboxes, dtype=np.float32)
        return bboxes

    def fill_template(self, img, content, imgdatakey):
        #

        jsonimg = self.data_template
        #
        jsonimg['_id'] = imgdatakey
        jsonimg['date'] = str(datetime.now())
        jsonimg['description']['format'] =  img.format
        jsonimg['description']['imageHeight'] = img.height
        jsonimg['description']['imageWidth'] = img.width
        jsonimg['description']['imageChannels'] = img.layers
        # Encoding image
        jsonimg['imgdatakey'] = imgdatakey # self.img_pil_to_data(img)

        # checking the size of data
        jsonimg['description']['sizeof'] = img.__sizeof__()

        # if image size is larger than a threshold to compress it
        """         {
            "attribute": {"color": "black",
                          "viewpoint": "rear",
                          "mark": "BMW",
                          "type": "SUV"
                          },
            #
            "label": "plate",
            "shape": {
                "bbox": {"xcenter": 0, "ycenter": 0, "width": 0, "height": 0},
                "points": [],
                "shape_type": "polygon",
                "landmark": {},
            },
        """
        # filling the annotation part
        for shape in content['shapes']:
            # check if shape is empty
            tmp_object = {'shape':{},  'attribute': {} }
            #
            if len(shape.keys()) ==0:
                continue

            if shape['label'] == 'plate':
                tmp_object['attribute'] = {}
                tmp_object['label'] = 'plate'
                tmp_object['shape']['points'] = shape['points']
                tmp_object['shape']['shape_type'] = shape['shape_type']

            elif shape['label'] in ['glass', 'blur', 'window']:
                tmp_object['attribute'] = {}
                tmp_object['label'] = 'window'
                tmp_object['shape']['points'] = shape['points']
                tmp_object['shape']['shape_type'] = shape['shape_type']

            else:
                if shape['label'] in ['forward', 'front', 'back']:
                    tmp_object['label'] = 'right_light'
                    tmp_object['shape']['points'] = shape['points']
                    tmp_object['shape']['shape_type'] = shape['shape_type']

                    if shape['label'] == 'back':
                        tmp_object['attribute']['viewpoint'] = 'rear'
                    else:
                        tmp_object['attribute']['viewpoint'] = 'front'

                elif shape['label'] in colorList: # color list = ['black', 'white' vs ]
                    tmp_object['label'] = 'left_light'
                    tmp_object['shape']['points'] = shape['points']
                    tmp_object['shape']['shape_type'] = shape['shape_type']
                    tmp_object['attribute']['color'] = shape['label'].split(' ')[0] # checking other label included

                if shape['label'] in markList:
                    tmp_object['label'] = 'mark_model'

                else:
                    tmp_object['shape']['points']     = shape['points']
                    tmp_object['shape']['shape_type'] = shape['shape_type']
                    tmp_object['attribute']['mark']   = shape['label'].split(' ')[0] # checking other label included


            #
            # ------------------------------------
            #
            jsonimg['annotation']['objects'].append(tmp_object)


        return jsonimg


def load_with_pickle(pkl_path):
    with open( pkl_path ,"rb") as f:
        ret = pickle.load(f)
    return ret




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')

    # part 1 read existing data
    data_dir = "C:\\Users\\iolgun\\Desktop\\allCountries\\"

   # imgList = glob.glob( data_dir + "*.jpg")
    labelList = glob.glob( data_dir + "*.json")

    # dataset handler
    put2json = data2json()
    conn = mongoHandler("AllCountriesCar").geDB()
    # metadata = mongoHandler("metadata").geDB()
    # get the database
    cconn = gridfs.GridFS(conn)
    # conn.compsCar.insert_one({})

    comp_data = []
    # reading images and creating json files
    for item in labelList:
        # read json file
        jsfile = open(item, 'r')
        item = json.load(jsfile)

        # open image
        if not os.path.exists( data_dir + item['imagePath'] ):
            continue
        img = Image.open(data_dir + item['imagePath'])

        imgf = open( data_dir + item['imagePath'] , 'rb')

        # get the annotations
        # s = json.dumps(comp_data)
        # storing data in mongoDB
        imgdatakey = cconn.put( imgf )
        """
        Note: To read back the image
          img = cconn.get(imgdatakey)
          f = io.BytesIO()
          f.write(img.read())
          img_pil = PIL.Image.open(f)
          img_pil.show()
        """
        comp_data =  put2json.fill_template(img, item, imgdatakey)

        conn['denemee'].insert_one(comp_data)

        # put data into database


print('finished')




""" 
self.data_template = {

    "version": "1.0.0",
    "date": "",
    "dataName": "",
    "flags": {
        "data_type": "image",
        "datasetName": "compsCar"
    },

    "description": {
        "content": "vehicle",
        "format": "jpg",
        "imageHeight": 0,
        "imageWidth": 0,
        "imageChannels": 1,
        'sizeof': 0,
    },

    "annotation": {"flags": {"num_objects": 1,
                             "group_id": None,  # generate group id and describe the here
                             },
                   # attributes

                   "objects": [
                       # light
                       {
                           # for point like objects (lights etc.)
                           "attribute": {},
                           "label": "yes",
                           "shape": {
                               "bbox": {},
                               "points": [[0, 0]],
                               "shape-type": "point",
                               "landmark": {}
                           },
                           "characters": {}

                       },

                       # number plate
                       {
                           "attribute": {"color": "black",
                                         "view point": "rear",
                                         "mark": "BMW",
                                         "type": "SUV"
                                         },
                           #
                           "label": "plate",
                           "shape": {
                               "bbox": {"xcenter": 0, "ycenter": 0, "width": 0, "height": 0},
                               "points": [
                                   [
                                       2017.4193548387095,
                                       743.4408602150537
                                   ],
                                   [
                                       2015.4838709677417,
                                       781.9354838709677
                                   ],
                                   [
                                       2092.9032258064512,
                                       783.6559139784946
                                   ],
                                   [
                                       2095.9139784946233,
                                       744.9462365591397
                                   ]
                               ],
                               "shape_type": "polygon",
                               "landmark": {},
                           },

                           # if any char exists
                           "characters": {"0": {"char": "A", "pos": []}}
                       },
                       # windows
                       {
                           "label": "window",
                           "shape": {
                               "bbox": {"xcenter": 0, "ycenter": 0, "width": 0, "height": 0},
                               "points": [
                                   [
                                       2017.4193548387095,
                                       743.4408602150537
                                   ],
                                   [
                                       2015.4838709677417,
                                       781.9354838709677
                                   ],
                                   [
                                       2092.9032258064512,
                                       783.6559139784946
                                   ],
                                   [
                                       2095.9139784946233,
                                       744.9462365591397
                                   ]
                               ],
                               "shape_type": "polygon",
                               "landmark": {},
                           },
                           "characters": {"0": {"char": "A", "pos": []}}
                       },
                   ]  # objects closed
                   },  # annotation closed

    "sentiments": None,
    "description": {}
}
"""

