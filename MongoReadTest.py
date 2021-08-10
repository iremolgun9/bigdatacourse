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




def load_with_pickle(pkl_path):
    with open( pkl_path ,"rb") as f:
        ret = pickle.load(f)
    return ret




# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    # dataset handler
    client = MongoClient('127.0.0.1', 27017 )
    conn = client["ColorDataset"]
    # metadata = mongoHandler("metadata").geDB()
    # get the database
    cconn = gridfs.GridFS(conn)
    # conn.compsCar.insert_one({})

    comp_data = []
    #datadir = 'D:/randomdata/'
    # reading images and creating json files

    #dataidlist =  load_with_pickle(datadir + 'dataids.pickle')

    dataids = conn.fs.files.find()

    for item in range(22000):
        # read json file
        # get the annotations
        # s = json.dumps(comp_data)
        # storing data in mongoDB
        dicl = next( dataids )
        imgf = cconn.get( dicl['_id'] )
        f = io.BytesIO()
        f.write(imgf.read())
        img_pil = PIL.Image.open(f)

        print('file ', item, 'is red.')


        #comp_data.append( str(imgdatakey) )

        """
        Note: To read back the image
          img = cconn.get(imgdatakey)
          f = io.BytesIO()
          f.write(img.read())
          img_pil = PIL.Image.open(f)
          img_pil.show()
        """
        #comp_data =  put2json.fill_template(img, item, imgdatakey)

       # conn['denemee'].insert_one(comp_data)

        # put data into database
    # conn['compsCar'].insert_many(comp_data)

#with open( datadir + 'dataids.pickle', 'wb') as f:
#    pickle.dump( comp_data, f)


print('finished')






