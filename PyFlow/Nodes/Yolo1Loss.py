from PyFlow.Loss.Yolo1 import custom_loss
from ..Core import Node
from ..Core.AbstractGraph import *


## If else node
class Yolo1Loss(Node):
    def __init__(self, name, graph):
        super(Yolo1Loss, self).__init__(name, graph)
        self.Loss_pin = self.addOutputPin("Loss Function", DataTypes.Any)


    @staticmethod
    def pinTypeHints():
        return {'inputs': [], 'outputs': [DataTypes.Any]}

    @staticmethod
    def category():
        return 'Keras|function'

    def compute(self):

        self.Loss_pin.setData(custom_loss)

