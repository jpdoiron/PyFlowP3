import os

from PyFlow.Core.AGraphCommon import DataTypes
from ..Core import Node


class ConvertH5ToTfLite(Node):
    def __init__(self, name, graph):
        super(ConvertH5ToTfLite, self).__init__(name, graph)
        self.in0 = self.addInputPin('In', DataTypes.Exec, self.compute)
        self.completed_pin = self.addOutputPin('Completed', DataTypes.Exec)

        self.output_model_pin = self.addInputPin('output_model', DataTypes.String, self.compute)
        self.input_model_pin = self.addInputPin('input_model', DataTypes.String, self.compute)
        self.input_field_pin = self.addInputPin('input_field', DataTypes.String, self.compute)
        self.output_field_pin = self.addInputPin('output_field', DataTypes.String, self.compute)
        self.mean_pin = self.addInputPin('mean', DataTypes.String, self.compute)
        self.std_dev_pin = self.addInputPin('std_dev', DataTypes.String, self.compute)
        self.input_shape_pin = self.addInputPin('input_shape', DataTypes.String, self.compute)


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
        return ['convert', 'h5']

    @staticmethod
    def description():
        '''
            used by property view and node box widgets
        '''
        return 'waka waka'

    def compute(self):
        command = "tflite_convert \
        --output_file={output_file} \
        --graph_def_file={graph_def_file} \
        --input_arrays={input_arrays} \
        --output_arrays={output_arrays} \
        --mean_values={mean_values} \
        --input_shape={input_shape} \
        --std_dev_values={std_dev_values}".format(output_file=self.output_model_pin.getData(),
                                                  graph_def_file=self.input_model_pin.getData(),
                                                  input_arrays=self.input_field_pin.getData(),
                                                  output_arrays=self.output_field_pin.getData(),
                                                  mean_values=self.mean_pin.getData(),
                                                  std_dev_values=self.std_dev_pin.getData(),
                                                  input_shape=self.input_shape_pin.getData())

        print(command)
        os.system(command, )