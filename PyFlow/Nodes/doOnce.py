from ..Core import Node
from ..Core.AbstractGraph import *


class doOnce(Node):
    def __init__(self, name, graph):
        super(doOnce, self).__init__(name, graph)
        self.inExec = self.addInputPin('inExec', DataTypes.Exec, self.compute, hideLabel=True)
        self.reset = self.addInputPin('Reset', DataTypes.Exec, self.OnReset)
        self.bStartClosed = self.addInputPin('Start closed', DataTypes.Bool)
        self.completed = self.addOutputPin('Completed', DataTypes.Exec)
        self.bClosed = False

    def OnReset(self):
        self.bClosed = False
        self.bStartClosed.setData(False)

    @staticmethod
    def pinTypeHints():
        return {'inputs': [DataTypes.Exec, DataTypes.Bool], 'outputs': [DataTypes.Exec]}

    @staticmethod
    def category():
        return 'FlowControl'

    @staticmethod
    def keywords():
        return []

    @staticmethod
    def description():
        return 'Will fire off an execution pin just once. But can reset.'

    @threaded
    def compute(self):
        bStartClosed = self.bStartClosed.getData()

        if not self.bClosed and not bStartClosed:
            self.completed.call()
            self.bClosed = True
            self.bStartClosed.setData(False)
