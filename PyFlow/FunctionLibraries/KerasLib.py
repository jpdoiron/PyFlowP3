import random
import sys

from tensorflow.python.keras import layers
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.layers.base import Layer

from PyFlow.Core.AGraphCommon import DataTypes, NodeTypes
from PyFlow.Core.FunctionLibrary import FunctionLibraryBase, IMPLEMENT_NODE


class KerasLib(FunctionLibraryBase):
    '''doc string for KerasLib'''
    def __init__(self):
        super(KerasLib, self).__init__()

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|SubCategory name', 'Keywords': ['+', 'merge', 'concate']})
    def Merge(Layer_1=(DataTypes.Layer, Layer(0)), Layer_2=(DataTypes.Layer, Layer(0)),LayerName=(DataTypes.String, "")):
        '''Sum of two ints.'''
        if(LayerName==""):
            LayerName = "{}{}".format(sys._getframe().f_code.co_name,random.randint(0,1000))

        return layers.Concatenate(name=LayerName)([Layer_1, Layer_2])


    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.String, 0),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|SubCategory name', 'Keywords': ['build']})
    def Build(Input=(DataTypes.Layer, None), Layers=(DataTypes.Layer, None)):
        '''Sum of two ints.'''
        m= Model(Input, Layers)
        return m.summary()

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|SubCategory name', 'Keywords': ['Normalization',"batch"]})
    def BatchNormalization(Input=(DataTypes.Layer, Layer(0)),LayerName=(DataTypes.String, "")):
        '''Sum of two ints.'''
        if(LayerName==""):
            LayerName = "{}{}".format(sys._getframe().f_code.co_name,random.randint(0,1000))

        return layers.BatchNormalization(name=LayerName)(Input)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|SubCategory name', 'Keywords': ['Activation']})
    def Activation(Input=(DataTypes.Layer, Layer(0)), Type=(DataTypes.String, "relu"),LayerName=(DataTypes.String, "")):
        '''Sum of two ints.'''
        if(LayerName==""):
            LayerName = "{}{}".format(sys._getframe().f_code.co_name,random.randint(0,1000))

        return layers.Activation(activation=Type,name=LayerName)(Input)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|SubCategory name', 'Keywords': ['AveragePooling2D',"pool"]})
    def MaxPooling2D(Input=(DataTypes.Layer, Layer(0)),pool_size=(DataTypes.Int, 32),strides=(DataTypes.Int, 1),padding=(DataTypes.String, "same"),LayerName=(DataTypes.String, "")):
        '''Sum of two ints.'''
        # if(LayerName==""):
        #     LayerName = "{}{}".format(sys._getframe().f_code.co_name,random.randint(0,1000))

        return layers.MaxPooling2D(pool_size=pool_size,strides=strides,padding=padding, name=LayerName)(Input)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|SubCategory name', 'Keywords': ['AveragePooling2D',"pool"]})
    def AveragePooling2D(Input=(DataTypes.Layer, Layer(0)),pool_size=(DataTypes.Int, 32),strides=(DataTypes.Int, 1),padding=(DataTypes.String, "same"),LayerName=(DataTypes.String, "")):
        '''Sum of two ints.'''
        if(LayerName==""):
            LayerName = "{}{}".format(sys._getframe().f_code.co_name,random.randint(0,1000))

        return layers.AveragePooling2D(pool_size=pool_size,strides=strides,padding=padding, name=LayerName)(Input)


    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|SubCategory name', 'Keywords': ['SeparableConv2D',"Conv"]})
    def SeparableConv2D(Input=(DataTypes.Layer, Layer(0)),filters=(DataTypes.Int, 128),kernel_size=(DataTypes.Int, 32),strides=(DataTypes.Int, 1),padding=(DataTypes.String, "same"),LayerName=(DataTypes.String, "")):
        '''Sum of two ints.'''
        # if(LayerName==""):
        #     LayerName = "{}{}".format(sys._getframe().f_code.co_name,random.randint(0,1000))
        #
        return layers.SeparableConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=LayerName)(Input)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|SubCategory name', 'Keywords': ['SeparableConv2D',"Conv"]})
    def Conv2D(Input=(DataTypes.Layer, Layer(0)),filters=(DataTypes.Int, 128),kernel_size=(DataTypes.Int, 32),strides=(DataTypes.Int, 1),padding=(DataTypes.String, "same"),LayerName=(DataTypes.String, "")):
        '''Sum of two ints.'''
        # if(LayerName==""):
        #     LayerName = "{}{}".format(sys._getframe().f_code.co_name,random.randint(0,1000))
        #
        return layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=LayerName)(Input)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|SubCategory name', 'Keywords': ['Dense']})
    def Dense(Input=(DataTypes.Layer, Layer(0)), units=(DataTypes.Int, 1024),Type=(DataTypes.String, "relu"),LayerName=(DataTypes.String, "")):
        '''Sum of two ints.'''
        # if(LayerName==""):
        #     LayerName = "{}{}".format(sys._getframe().f_code.co_name,random.randint(0,1000))

        return layers.Dense(units=units, activation=Type,name=LayerName)(Input)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|SubCategory name', 'Keywords': ['Flatten']})
    def Flatten(Input=(DataTypes.Layer, Layer(0)), LayerName=(DataTypes.String, "")):
        '''Sum of two ints.'''
        # if(LayerName==""):
        #     LayerName = "{}{}".format(sys._getframe().f_code.co_name,random.randint(0,1000))

        return layers.Flatten(name=LayerName)(Input)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None),nodeType=NodeTypes.Callable, meta={'Category': 'Keras|SubCategory name', 'Keywords': ['Flatten']})
    def Input(input_size=(DataTypes.Int, 224),input_channel=(DataTypes.Int, 3), LayerName=(DataTypes.String, "")):
        '''Sum of two ints.'''
        # if(LayerName==""):
        #     LayerName = "{}{}".format(sys._getframe().f_code.co_name,random.randint(0,1000))

        return layers.Input(shape=(input_size, input_size, input_channel),name=LayerName)

    #output = tf.keras.layers.Input(shape=(input_size, input_size, input_channel), name="input")

