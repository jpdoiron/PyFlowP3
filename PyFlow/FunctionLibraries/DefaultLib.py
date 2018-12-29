import os
from datetime import time

from ..Core.FunctionLibrary import *


#TODO jpd rempve duplicate function
class DefaultLib(FunctionLibraryBase):
    '''
    Default library builting stuff, variable types and conversions
    '''
    def __init__(self):
        super(DefaultLib, self).__init__()

    @staticmethod
    @IMPLEMENT_NODE(returns=None, nodeType=NodeTypes.Callable, meta={'Category': 'DefaultLib', 'Keywords': ['print']})
    ## Python's 'print' function wrapper
    def pyprint(entity=(DataTypes.String, None)):
        '''
        printing a string
        '''
        print(entity)

    @staticmethod
    @IMPLEMENT_NODE(returns=None, nodeType=NodeTypes.Callable, meta={'Category': 'DefaultLib', 'Keywords': []})
    ## cls cmd call.
    def cls():
        '''cls cmd call.'''
        os.system('cls')

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Bool, False), meta={'Category': 'Math|Bool', 'Keywords': []})
    ## make boolean
    def makeBool(b=(DataTypes.Bool, False)):
        return b

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Int, 0), meta={'Category': 'GenericTypes', 'Keywords': []})
    ## make integer
    def makeInt(i=(DataTypes.Int, 0)):
        '''make integer'''
        return i

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'GenericTypes', 'Keywords': []})
    ## make floating point number
    def makeFloat(f=(DataTypes.Float, 0.0)):
        '''make floating point number'''
        return f

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.String, ''), meta={'Category': 'GenericTypes', 'Keywords': []})
    ## make string
    def makeString(s=(DataTypes.String, '')):
        '''make string'''
        return s

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Any, 0,{"constraint":"1"}), meta={'Category': 'Conversion', 'Keywords': []})
    def passtrhough(input=(DataTypes.Any, 0,{"constraint":"1"})):
        return input

    # Conversions
    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Bool, False), meta={'Category': 'Conversion', 'Keywords': []})
    def intToBool(i=(DataTypes.Int, 0)):
        return bool(i)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Int, 0), meta={'Category': 'Conversion', 'Keywords': []})
    def floatToInt(f=(DataTypes.Float, 0.0)):
        return int(f)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, 0.0), meta={'Category': 'Utils', 'Keywords': []})
    ## Returns the CPU time or real time since the start of the process or since the first call of clock()
    def clock():
        '''Returns the CPU time or real time since the start of the process or since the first call of clock().'''
        return time.clock()

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, False), meta={'Category': 'Conversion', 'Keywords': []})
    def intToFloat(i=(DataTypes.Int, 0)):
        return float(i)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.String, ''), meta={'Category': 'Conversion', 'Keywords': []})
    def intToString(i=(DataTypes.Int, 0)):
        return str(i)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.String, ''), meta={'Category': 'Conversion', 'Keywords': []})
    def floatToString(f=(DataTypes.Float, 0.0)):
        return str(f)


    # Conversions
    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Bool, False), meta={'Category': 'Conversion', 'Keywords': ["Bool"]})
    def toBool(i=(DataTypes.Any, 0,{"supportedDataTypes":[DataTypes.Bool,DataTypes.Float,DataTypes.Int]})):
        return bool(i)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Int, 0), meta={'Category': 'Conversion', 'Keywords': []})
    def toInt(i=(DataTypes.Any, 0,{"supportedDataTypes":[DataTypes.Bool,DataTypes.Float,DataTypes.Int]})):
        return int(i)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.Float, False), meta={'Category': 'Conversion', 'Keywords': []})
    def toFloat(i=(DataTypes.Any, 0,{"supportedDataTypes":[DataTypes.Bool,DataTypes.Float,DataTypes.Int]})):
        return float(i)

    @staticmethod
    @IMPLEMENT_NODE(returns=(DataTypes.String, ''), meta={'Category': 'Conversion', 'Keywords': []})
    def toString(i=(DataTypes.Any, 0)):
        return str(i)

