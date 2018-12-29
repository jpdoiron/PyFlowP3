from PyFlow.Core.Settings import Colors
from ..Core.AGraphCommon import *
from ..Core.Pin import PinWidgetBase


class IntPin(PinWidgetBase):
    """doc string for IntPin"""
    def __init__(self, name, parent, dataType, direction, **kwargs):
        super(IntPin, self).__init__(name, parent, dataType, direction, **kwargs)
        self.setDefaultValue(0)

    @staticmethod
    def color():
        return Colors.Int

    @staticmethod
    def pinDataTypeHint():
        return DataTypes.Int, 0

    def supportedDataTypes(self):
        return (DataTypes.Int, DataTypes.Float)

    def setData(self, data):
        try:
            self._data = int(data)
        except:
            self._data = self.defaultValue()
        PinWidgetBase.setData(self, self._data)