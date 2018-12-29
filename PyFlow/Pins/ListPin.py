from PyFlow.Core.Settings import Colors
from ..Core.AGraphCommon import *
from ..Core.Pin import PinWidgetBase


class ListPin(PinWidgetBase):
    """doc string for ListPin"""
    def __init__(self, name, parent, dataType, direction, **kwargs):
        super(ListPin, self).__init__(name, parent, dataType, direction, **kwargs)
        self.setDefaultValue([])

    def supportedDataTypes(self):
        return (DataTypes.Array,)

    @staticmethod
    def color():
        return Colors.Array

    @staticmethod
    def pinDataTypeHint():
        return DataTypes.Array, []

    def setData(self, data):
        if isinstance(data, list):
            self._data = data
        else:
            self._data = self.defaultValue()
        PinWidgetBase.setData(self, self._data)
