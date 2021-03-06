from PyFlow.Core.Settings import Colors
from ..Core.AGraphCommon import *
from ..Core.Pin import PinWidgetBase


class FloatPin(PinWidgetBase):
    """doc string for FloatPin"""
    def __init__(self, name, parent, dataType, direction, **kwargs):
        super(FloatPin, self).__init__(name, parent, dataType, direction, **kwargs)
        self.setDefaultValue(0.0)

    def supportedDataTypes(self):
        return (DataTypes.Float, DataTypes.Int)

    @staticmethod
    def color():
        return Colors.Float

    @staticmethod
    def pinDataTypeHint():
        return DataTypes.Float, 0.0

    def setData(self, data):
        try:
            self._data = float(data)
        except:
            self._data = self.defaultValue()
        PinWidgetBase.setData(self, self._data)
