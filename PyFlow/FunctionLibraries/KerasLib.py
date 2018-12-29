from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.layers.base import Layer

from PyFlow.Core.AGraphCommon import DataTypes
from PyFlow.Core.FunctionLibrary import FunctionLibraryBase, IMPLEMENT_NODE


class KerasLib(FunctionLibraryBase):
    '''doc string for KerasLib'''
    def __init__(self):
        super(KerasLib, self).__init__()

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Layer, None), meta={'Category': 'Keras|SubCategory name', 'Keywords': ['+', 'merge', 'concate']})
    def Merge(A=(DataTypes.Layer, Layer(0)), B=(DataTypes.Layer, Layer(0))):
        '''Sum of two ints.'''
        return Concatenate()([A,B])

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.String, 0), meta={'Category': 'Keras|SubCategory name', 'Keywords': ['build']})
    def Print(A=(DataTypes.Layer, 0),B=(DataTypes.Layer, 0)):
        '''Sum of two ints.'''
        return "ALLO"
        # m= Model(A,B)
        # m.summary()
        # return m.summary()


    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.String, 0), meta={'Category': 'Keras|SubCategory name', 'Keywords': ['build']})
    def Build(A=(DataTypes.Layer, None),B=(DataTypes.Layer, None)):
        '''Sum of two ints.'''
        m= Model(A,B)
        m.summary()
        return m.summary()

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'CategoryName', 'Keywords': ['/']})
    def divide(A=(DataTypes.Int, 0), B=(DataTypes.Int, 0), result=(DataTypes.Reference, (DataTypes.Bool, False))):
        '''Integer devision.'''
        try:
            d = A / B
            result(True)
            return d
        except:
            result(False)
            return -1

