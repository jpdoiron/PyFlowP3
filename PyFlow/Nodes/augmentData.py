from Utils.augmentation import augment_image
from ..Core import Node
from ..Core.AbstractGraph import *


class augmentData(Node):
    def __init__(self, name, graph):
        super(augmentData, self).__init__(name, graph)

        self.image_pin = self.addInputPin('image', DataTypes.Any, defaultValue=None)
        self.annotation_pin = self.addInputPin('annotation', DataTypes.Array,defaultValue=[])

        self.augmenter_pin = self.addOutputPin('Augmenter', DataTypes.Any)




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
        return 'Keras|function'

    @staticmethod
    def keywords():
        '''
            used by nodebox filter while typing
        '''
        return ['Augmentation','label']


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

    @threaded
    def compute(self):

        try:

            image = self.image_pin.getData()
            annotation = self.annotation_pin.getData()

            if image!= None:
                img, frame = augment_image(image, annotation[:4], self.log_path if self.debug_Augmentation > 0 else "")
                print("augmentation:" , img,frame)
            self.augmenter_pin.setData(augment_image)

        except Exception as e:
            import traceback
            import sys
            traceback.print_exception(type(e), e, sys.exc_info()[2], limit=1, file=sys.stdout)






