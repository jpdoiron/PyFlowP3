from Utils.image_utils import draw_detection_on_image
from Utils.yolo1.utils import detect_image
from ..Core import Node
from ..Core.AbstractGraph import *


class detectImage(Node):
    def __init__(self, name, graph):
        super(detectImage, self).__init__(name, graph)
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

    @threaded
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

            out_classes, out_boxes, out_scores, transform, exec_time, _ = detect_image(image, model)
            image = draw_detection_on_image(self, image, out_boxes, out_classes, out_scores)

            import matplotlib.pyplot as plt

            # pred = self.detect(image)
            # draw = self.GetBox(pred, img, dim=img.shape[:2])

            #fixme called from thread
            #plt.imshow(image[..., ::-1])
            #plt.show()

            print("RESULT de la mort", out_boxes)

            self.out0.call()

        except Exception as e:
            import traceback
            import sys
            traceback.print_exception(type(e), e, sys.exc_info()[2], limit=1, file=sys.stdout)

