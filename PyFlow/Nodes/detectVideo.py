from timeit import default_timer as timer

import cv2
import numpy as np

from Utils.image_utils import draw_detection_on_image, image_resize2, transform_box
from Utils.yolo1 import utils
from ..Core import Node
from ..Core.AbstractGraph import *


class detectVideo(Node):
    def __init__(self, name, graph):
        super(detectVideo, self).__init__(name, graph)
        self.inp0 = self.addInputPin('in0', DataTypes.Exec,self.compute)

        self.model_pin = self.addInputPin('model', DataTypes.Any)
        self.videoPath_pin = self.addInputPin('videoPath', DataTypes.Any, defaultValue=-1)


        self.out0 = self.addOutputPin('out0', DataTypes.Exec)
        pinAffects(self.inp0, self.out0)


    @staticmethod
    def pinTypeHints():
        '''
            used by nodebox to suggest supported pins
            when drop wire from pin into empty space
        '''
        return {'inputs': [DataTypes.String], 'outputs': [DataTypes.Any]}

    @staticmethod
    def category():
        '''
            used by nodebox to place in tree
            to make nested one - use '|' like this ( 'CatName|SubCatName' )
        '''
        return 'Common'

    @staticmethod
    def keywords():
        '''
            used by nodebox filter while typing
        '''
        return []

    @staticmethod
    def description():
        '''
            used by property view and node box widgets
        '''
        return 'default description'


    #todo remove duplicate from detectVideo and detecImage, maybe a new pin
    def ProcessInput(self, image):
        from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
        '''image channel transformation for training / inferencing'''
        # b, g, r = cv2.split(image)  # get b,g,r
        # image = cv2.merge([r, g, b])  # switch it to rgb
        image = preprocess_input(image, mode='tf')
        return image


    def detect_image(self, image_in, model, center=(0, 0)):
        image_dims = image_in.shape
        image, transform = image_resize2(image_in,(224,224), center=center)

        image = self.ProcessInput(image)

        image_data = np.expand_dims(image, axis=0)  # Add batch dimension.


        start = timer()
        pred = model.predict(image_data)
        end = timer()
        time = (end - start)

        #todo those utils are specific to YOLO1, need to param them
        bboxes1 = utils.get_boxes(pred[0], cutoff=0.3)
        bboxes = utils.nonmax_suppression(bboxes1, iou_cutoff=0.05)

        out_boxes = []
        out_scores = []
        out_classes = []

        for bbox in bboxes:
            (x, y), (x1, y1), conf, cl = bbox
            box = transform_box((x,y,x1,y1),transform, inverse=True)
            out_boxes.append(box)
            out_scores.append(conf)
            out_classes.append(cl)

        return out_classes, out_boxes, out_scores, transform, time, image



    def detect_video(self, video_path):
        vid = cv2.VideoCapture(0)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")

        # if isOutput:
        #     print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        #     out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video/_size)
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        while True:
            return_value, frame = vid.read()
            height, width, _ = image.shape

            #todo detect image is model type specific
            out_classes, out_boxes, out_scores, transform, exec_time, _ = self.detect_image(image, center=(width // 2, height // 2))
            image = draw_detection_on_image(self, frame, out_boxes, out_classes, out_scores)

            result = np.asarray(image)
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            #if isOutput:
            #    out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('result', 0) == 0:
                break


        self.close_session()


    def compute(self):
        '''
            1) get data from inputs
            2) do stuff
            3) put data to outputs
            4) call output execs
        '''

        model= self.model_pin.getData()
        videoPath= self.videoPath_pin.getData()

        try:

            self.detect_video(videoPath)

            self.out0.call()


        except Exception as e:
            import traceback
            import sys
            traceback.print_exception(type(e), e, sys.exc_info()[2], limit=1, file=sys.stdout)

