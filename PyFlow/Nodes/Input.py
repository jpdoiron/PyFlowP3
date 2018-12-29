import tensorflow as tf

from PyFlow.Core.AGraphCommon import DataTypes
from PyFlow.Core.Node import Node


class Input(Node):
    def __init__(self, name, graph):
        super(Input, self).__init__(name, graph)
        self.inp0 = self.addInputPin('in0', DataTypes.Exec, self.compute, hideLabel=True)
        self.completed = self.addOutputPin('completed', DataTypes.Exec)
        self.input_pin = self.addInputPin('input1', DataTypes.Int)
        self.filter_pin = self.addInputPin('Filter', DataTypes.Int)
        self.kernel_pin = self.addInputPin('Kernel Size', DataTypes.Int)
        self.name_pin= self.addInputPin('Name', DataTypes.String)

        self.out0 = self.addOutputPin('out0', DataTypes.Layer)
        # pinAffects(self.input_pin, self.out0)

    @staticmethod
    def pinTypeHints():
        '''
            used by nodebox to suggest supported pins
            when drop wire from pin into empty space
        '''
        return {'inputs': [DataTypes.Int], 'outputs': [DataTypes.Layer]}

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

        input_data = self.input_pin.getData()
        kernel = self.kernel_pin.getData()
        filter = self.filter_pin.getData()
        layername= self.name_pin.getData()


        output = tf.keras.layers.Input(shape=(None, None, 3))

        try:
            self.out0.setData(output)
        except Exception as e:
            print(e)

        print(type(self).__name__)
        self.completed.call()

