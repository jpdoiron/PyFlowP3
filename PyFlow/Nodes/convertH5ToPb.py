import os
from pathlib import Path

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope

from PyFlow.Core.AGraphCommon import DataTypes
from ..Core import Node


def relu6(x):
    return K.relu(x, max_value=6)

class convertH5ToPb(Node):
    def __init__(self, name, graph):
        super(convertH5ToPb, self).__init__(name, graph)
        self.in0 = self.addInputPin('In', DataTypes.Exec, self.compute)
        self.completed_pin = self.addOutputPin('Completed', DataTypes.Exec)

        self.output_model_file_pin = self.addInputPin('output_model', DataTypes.String, self.compute, defaultValue="model.pb")
        self.output_folder_pin = self.addInputPin('output_folder', DataTypes.String, self.compute, defaultValue="")
        self.input_model_pin = self.addInputPin('input_model', DataTypes.Any, self.compute, defaultValue=None)

        self.theano_backend_pin = self.addInputPin('theano', DataTypes.Bool, self.compute, defaultValue=False)
        self.graph_def_pin = self.addInputPin('graph_def', DataTypes.Bool, self.compute, defaultValue=False)

        self.graph_def_file_pin = self.addInputPin('graph_def_file', DataTypes.String, self.compute, defaultValue="")
        self.num_output_pin = self.addInputPin('num_output', DataTypes.Int, self.compute, defaultValue=1)
        self.output_node_prefix_pin = self.addInputPin('output_node_prefix', DataTypes.String, self.compute, defaultValue="output_node")

        self.quantize_pin = self.addInputPin('quantize', DataTypes.Bool, self.compute, defaultValue=False)

        self._model_out_pin = self.addOutputPin('model_out', DataTypes.String, self.compute, defaultValue="")

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
        return 'convert h5 to Pb'

    def compute(self):
        """
        Convert H5 to PB
        """
        try:

            output_fld = self.output_folder_pin.getData()
            if len(output_fld) > 0:
                Path(output_fld).mkdir(parents=True, exist_ok=True)
            else:
                output_fld = os.getcwd()

            with CustomObjectScope({'relu6': relu6}):
                K.set_learning_phase(0)
                if self.theano_backend_pin.getData():
                    K.set_image_data_format('channels_first')
                else:
                    K.set_image_data_format('channels_last')

                try:
                    #model
                    net_model = self.input_model_pin.getData()
                except ValueError as err:
                    raise err
                num_output = self.num_output_pin.getData()
                pred = [None] * num_output
                pred_node_names = [None] * num_output
                for i in range(num_output):
                    pred_node_names[i] = self.output_node_prefix_pin.getData() + str(i)
                    pred[i] = tf.identity(net_model.outputs[i], name=pred_node_names[i])
                print('output nodes names are: ', pred_node_names)

                # [optional] write graph definition in ascii

                # In[ ]:

                sess = K.get_session()

                if self.graph_def_pin.getData():
                    f = self.graph_def_file_pin.getData()
                    tf.train.write_graph(sess.graph.as_graph_def(), output_fld, f, as_text=True)
                    print('saved the graph definition in ascii format at: ', str(Path(output_fld) / f))

                    # convert variables to constants and save

                    # In[ ]:

                from tensorflow.python.framework import graph_util
                from tensorflow.python.framework import graph_io
                if self.quantize_pin.getData():
                    from tensorflow.tools.graph_transforms import TransformGraph
                    transforms = ["quantize_weights", "quantize_nodes"]
                    transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], pred_node_names, transforms)
                    constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, pred_node_names)
                else:
                    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(),
                                                                               pred_node_names)
                graph_io.write_graph(constant_graph, output_fld, self.output_model_file_pin.getData(), as_text=False)

                self._model_out_pin.setData(self.output_model_file_pin.getData())
                print('saved the freezed graph (ready for inference) at: ', str(Path(output_fld) / self.output_model_file_pin.getData()))

        except Exception as e:
            import traceback
            import sys
            traceback.print_exception(type(e), e, sys.exc_info()[2], limit=1, file=sys.stdout)

