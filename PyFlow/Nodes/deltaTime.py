from ..Core import Node
from ..Core.AbstractGraph import *


class deltaTime(Node):
    def __init__(self, name, graph):
        super(deltaTime, self).__init__(name, graph)
        self._deltaTime = 0.0
        self._out0 = self.addOutputPin('out0', DataTypes.Float, hideLabel=True)

    @staticmethod
    def pinTypeHints():
        return {'inputs': [], 'outputs': [DataTypes.Float]}

    @staticmethod
    def category():
        return 'Utils'

    @staticmethod
    def keywords():
        return []

    @staticmethod
    def description():
        return 'Editor delta time.'

    def Tick(self, deltaTime):
        self._deltaTime = deltaTime

    @threaded
    def compute(self):
        self._out0.setData(self._deltaTime)
