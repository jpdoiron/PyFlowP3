import os
from multiprocessing.dummy import Pool as ThreadPool

import cv2
import numpy as np
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.callbacks import LambdaCallback

from PyFlow.Loss.Yolo1 import label_to_tensor
from Utils.augmentation import augment_image
from Utils.image_utils import image_resize2, transform_box
from ..Core import Node
from ..Core.AbstractGraph import *


class generator(Node):
    def __init__(self, name, graph):
        super(generator, self).__init__(name, graph)

        self.dataset_pin = self.addInputPin('Annotations', DataTypes.Array, defaultValue=[])
        self.valDataset_pin = self.addInputPin('Val annotations', DataTypes.Array, defaultValue=[])
        self.dataPath_pin = self.addInputPin('data path', DataTypes.String, defaultValue=None)

        self.augmenter_pin = self.addInputPin('Augmenter', DataTypes.Any,defaultValue=None)
        self.batchSize_pin = self.addInputPin('batch size', DataTypes.Int,defaultValue=0)
        self.logPath_pin = self.addInputPin('log path', DataTypes.String, defaultValue=None)

        self.generator_pin = self.addOutputPin('generator', DataTypes.Any)
        self.valGenerator_pin = self.addOutputPin('val generator', DataTypes.Any)

        self.threadpool = ThreadPool(32)


        #pinAffects(self.in0, self.completed_pin)

    @staticmethod
    def pinTypeHints():
        '''
            used by nodebox to suggest supported pins
            when drop wire from pin into empty space
        '''
        return {'inputs': [], 'outputs': []}



    @staticmethod
    def category():
        '''
            used by nodebox to place in tree
            to make nested one - use '|' like this ( 'CatName|SubCatName' )
        '''
        return 'Keras | function'

    @staticmethod
    def keywords():
        '''
            used by nodebox filter while typing
        '''
        return ['generator','label']


    @staticmethod
    def description():
        '''
            used by property view and node box widgets
        '''
        return 'default description'

    def serialize(self):
        template = Node.serialize(self)
        # if hasattr(template["value"], '__class__'):
        #     template['value'] = None
        for i in list(template["outputs"])+template["inputs"]:
            i["value"] = None

        return template


    from cachetools import cached, Cache

    cache = Cache(maxsize=30000)

    @cached(cache)
    def get_image(self, filename, imgsize, center):
        image = cv2.imread(filename)
        image, trans = image_resize2(image, imgsize, center=center)
        return image, trans


    def processFiles(self, label, imgsize=(224, 224)):

        #self.data_path = "D:/dev/data/MattelCars/2cars"
        #self.log_path  = "/logs"
        #self.debug_Augmentation = 0
        #self.augmentation = False

        """
        Takes the image file name and the frame (rows corresponding to a single image in the labels.csv)
        and randomly scales, translates, adjusts SV values in HSV space for the image,
        
        and adjusts the coordinates in the 'frame' accordingly, to match bounding boxes in the new image
        """
        ori_label, ori_frame = label.split()

        frame = np.array(list(map(int, ori_frame.split(','))))

        x,y,x1,y1 = frame[:4]
        center = ((x+x1)//2,(y+y1)//2)
        filename = os.path.join(self.data_path, ori_label)
        img, trans = self.get_image(filename, imgsize, center)

        frame[:4] = transform_box(frame[:4], trans, imgsize)

        #show_box(img,frame[:4])
        if self.augmenter != None:
            #self.debug_Augmentation = self.debug_Augmentation - 1
            #img, frame_ret= augment_image(img,frame[:4],self.log_path if self.debug_Augmentation>0 else "")
            img, frame_ret= self.augmenter(img,frame[:4])
            frame = np.append(frame_ret[:4], frame[4])
            #frame[:4] = ClampBox(frame[:4], imgsize)

        #show_box(img, frame[:4])
        frame_tensor=[]
        if (len(frame) > 1):
            frame_tensor = label_to_tensor(frame)

        img = self.ProcessInput(img)
        return img, frame_tensor



    def ProcessInput(self, image):
        '''image channel transformation for training / inferencing'''
        #b, g, r = cv2.split(image)  # get b,g,r
        #image = cv2.merge([r, g, b])  # switch it to rgb
        image = preprocess_input(image, mode='tf')
        return image


    def generatorThread(self,labels, batch_size):
        """
        Generator function
        # Arguments
        label_keys: image names, that are keys of the label_frames Arguments
        label_frames: array of frames (rows corresponding to a single image in the labels.csv)
        batch_size: batch size
        """
        n = len(labels)
        i=0

        while 1:

            if i + batch_size > n:
                np.random.shuffle(labels)
                i=0

            output = self.threadpool.starmap(self.processFiles, zip(labels[i:i + batch_size]))

            image_data , box_data = list(zip(*output))

            i = i + batch_size

            X_train = np.array(image_data)

            idx_to_remove = []
            y_train=[]
            isinit=False
            for idx,x in enumerate(box_data):
                if isinstance(x, list):
                    x = np.asarray(x)
                x = x.reshape((1,x.size))
                if x.size==0:
                    idx_to_remove.append(idx)
                    continue

                if (not isinit):
                    y_train = x
                    isinit = True
                else:
                    y_train = np.vstack((y_train,x))

            if(len(idx_to_remove)>0):
                X_train = np.delete(X_train, idx_to_remove, axis=0)

            # b = get_boxes(y_train[0],label=True)
            # (x, y), (x1, y1), _, _ = b[0]
            # a = np.copy(X_train[0])
            # show_box(a,[x,y,x1,y1])
            yield (X_train, y_train)



    def compute(self):

        print("fit gen")
        try:
            dataset = self.dataset_pin.getData()
            valDataset = self.valDataset_pin.getData()
            batchSize = self.batchSize_pin.getData()
            self.augmenter= self.augmenter_pin.getData()
            self.data_path= self.dataPath_pin.getData()
            self.log_path= self.logPath_pin.getData()

            if (len(dataset) > 0):
                self.generator_pin.setData(self.generatorThread(dataset,batchSize))

            if (len(valDataset) > 0):
                self.valGenerator_pin.setData(self.generatorThread(dataset,batchSize))

        except Exception as e:
            import traceback
            import sys
            traceback.print_exception(type(e), e, sys.exc_info()[2], limit=1, file=sys.stdout)






