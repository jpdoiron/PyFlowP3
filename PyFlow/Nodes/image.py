import cv2
from PySide2.QtWidgets import QGraphicsItem

from ..Core import Node
from ..Core.AbstractGraph import *


class image(Node):
    def __init__(self, name, graph):
        super(image, self).__init__(name, graph)
        self.imageName_pin = self.addInputPin('image name', DataTypes.String)
        self.dataPath_pin = self.addInputPin('image path', DataTypes.String)
        self.image_pin = self.addOutputPin('image', DataTypes.Any)

        self.imageBox = self.addImage()

    def itemChange(self, change, value):
        if change == self.ItemSelectedHasChanged:
            try:
                self.refreshImage()
            except:
                pass
            finally:
                return QGraphicsItem.itemChange(self, change, value)
        return QGraphicsItem.itemChange(self, change, value)

    def refreshImage(self):
        try:
            dataPath = self.dataPath_pin.getData()
            imageName = self.imageName_pin.getData()
            image = cv2.imread(dataPath + '/' + imageName)
            self.image_pin.setData(image)
            # a = self.imageBox.size()
            self.changeImage(self.imageBox, image, default = True if image is None else False)
        except:
            print("error loading image")
            pass

    def postCreate(self, jsonTemplate):
        Node.postCreate(self, jsonTemplate)
        self.refreshImage()

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

    @threaded
    def compute(self):
        '''
            1) get data from inputs
            2) do stuff
            3) put data to outputs
            4) call output execs
        '''

        try:

            self.refreshImage()

        except Exception as e:
            import traceback
            import sys
            traceback.print_exception(type(e), e, sys.exc_info()[2], limit=1, file=sys.stdout)

