import os

from PyFlow.Core.AGraphCommon import DataTypes
from ..Core import Node


class convertPbToTfLite(Node):
    def __init__(self, name, graph):
        super(convertPbToTfLite, self).__init__(name, graph)
        self.in0 = self.addInputPin('In', DataTypes.Exec, self.compute)
        self.completed_pin = self.addOutputPin('Completed', DataTypes.Exec)

        self.output_model_pin = self.addInputPin('output_model', DataTypes.String, self.compute, defaultValue="model.tflite")
        self.input_model_pin = self.addInputPin('input_model', DataTypes.String, self.compute, defaultValue="model.pb")
        self.input_field_pin = self.addInputPin('input_field', DataTypes.String, self.compute)
        self.output_field_pin = self.addInputPin('output_field', DataTypes.String, self.compute)
        self.mean_pin = self.addInputPin('mean', DataTypes.Int, self.compute, defaultValue=128)
        self.std_dev_pin = self.addInputPin('std_dev', DataTypes.Int, self.compute, defaultValue=127)
        self.input_shape_pin = self.addInputPin('input_shape', DataTypes.String, self.compute, defaultValue="")


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
        return 'convert H5 model to tflite'

    def compute(self):

        try:
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

        except Exception as e:
            import traceback
            import sys
            traceback.print_exception(type(e), e, sys.exc_info()[2], limit=1, file=sys.stdout)
