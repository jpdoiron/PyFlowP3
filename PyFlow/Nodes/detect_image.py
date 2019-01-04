
from Utils.yolo1 import utils
from ..Core.AbstractGraph import *
from ..Core.Settings import *
from Utils.image_utils import image_resize2, transform_box, draw_detection_on_image
from ..Core import Node
import numpy as np
from timeit import default_timer as timer

class detect_image(Node):
    def __init__(self, name, graph):
        super(detect_image, self).__init__(name, graph)
        self.inp0 = self.addInputPin('in0', DataTypes.Exec,self.compute)

        self.image_pin = self.addInputPin('image', DataTypes.Any)
        self.model_pin = self.addInputPin('model', DataTypes.Any)


        self.out0 = self.addOutputPin('out0', DataTypes.Exec)
        pinAffects(self.inp0, self.out0)

    @staticmethod
    def pinTypeHints():
        '''
            used by nodebox to suggest supported pins
            when drop wire from pin into empty space
        '''
        return {'inputs': [DataTypes.Exec,DataTypes.Any], 'outputs': [DataTypes.Exec]}

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

    def compute(self):
        '''
            1) get data from inputs
            2) do stuff
            3) put data to outputs
            4) call output execs
        '''
        try:
            image = self.image_pin.getData()
            model = self.model_pin.getData()

            out_classes, out_boxes, out_scores, transform, exec_time, _ = self.detect_image(image, model)
            image = draw_detection_on_image(self, image, out_boxes, out_classes, out_scores)

            import matplotlib.pyplot as plt

            # pred = self.detect(image)
            # draw = self.GetBox(pred, img, dim=img.shape[:2])
            plt.imshow(image[..., ::-1])
            plt.show()
            print("RESULT de la mort", out_boxes)

            self.out0.call()

        except Exception as e:
            import traceback
            import sys
            traceback.print_exception(type(e), e, sys.exc_info()[2], limit=1, file=sys.stdout)

