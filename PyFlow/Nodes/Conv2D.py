from tensorflow.python.layers import layers

from PyFlow.Core.AGraphCommon import DataTypes
from PyFlow.Core.Node import Node


class Conv2D(Node):
    def __init__(self, name, graph):
        super(Conv2D, self).__init__(name, graph)
        self.inp0 = self.addInputPin('in0', DataTypes.Exec,self.compute, hideLabel=True)
        self.completed = self.addOutputPin('completed', DataTypes.Exec)
        self.input_pin = self.addInputPin('input', DataTypes.Layer)
        self.filter_pin = self.addInputPin('Filter', DataTypes.Int)
        self.kernel_pin = self.addInputPin('Kernel Size', DataTypes.Int)
        self.name_pin= self.addInputPin('Name', DataTypes.String)

        self.out0 = self.addOutputPin('out0', DataTypes.Layer)
        # pinAffects(self.input, self.out0)

    @staticmethod
    def pinTypeHints():
        '''
            used by nodebox to suggest supported pins
            when drop wire from pin into empty space
        '''
        return {'inputs': [DataTypes.Layer], 'outputs': [DataTypes.Layer]}

    @staticmethod
    def category():
        '''
            used by nodebox to place in tree
            to make nested one - use '|' like this ( 'CatName|SubCatName' )
        '''
        return 'Keras'

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

    def compute(self):
        '''
            1) get data from inputs
            2) do stuff
            3) put data to outputs
            4) call output execs
        '''
        input1 = self.input_pin.getData()
        kernel1 = self.kernel_pin.getData()
        filter1 = self.filter_pin.getData()
        layername1= self.name_pin.getData()


        output = layers.Conv2D(filters=filter1, kernel_size=kernel1, name=layername1) (input1)

        try:
            self.out0.setData(output)
        except Exception as e:
            print(e)
        print(type(self).__name__)
        self.completed.call()
