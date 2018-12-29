from ..Core.AbstractGraph import *
from ..Core.Settings import *
from ..Core import Node


## charge node
#
# Each time node called it accumulates the step value.
# When accumulated value reaches "amount" - completed pin called.
# Useful when you need to wait some time inside some tick function.
class charge(Node):
    def __init__(self, name, graph):
        super(charge, self).__init__(name, graph)
        self.inExec = self.addInputPin('inExec', DataTypes.Exec, self.compute, hideLabel=True)
        self.amount = self.addInputPin('Amount', DataTypes.Float)
        self.amount.setDefaultValue(1.0)

        self.step = self.addInputPin('Step', DataTypes.Float)
        self.step.setDefaultValue(0.1)

        self.completed = self.addOutputPin('completed', DataTypes.Exec)
        self._currentAmount = 0

    @staticmethod
    def pinTypeHints():
        return {'inputs': [DataTypes.Float, DataTypes.Exec], 'outputs': [DataTypes.Exec]}

    @staticmethod
    def category():
        return 'FlowControl'

    @staticmethod
    def keywords():
        return []

    @staticmethod
    def description():
        return 'Each time node called it accumulates the step value. \
        When accumulated value reaches <b>"amount"</b> - completed pin called.\n\
        Useful when you need to wait some time inside some tick function.'

    def compute(self):
        step = abs(self.step.getData())
        if (self._currentAmount + step) < abs(self.amount.getData()):
            self._currentAmount += step
            return
        self.completed.call()
        self._currentAmount = 0.0
