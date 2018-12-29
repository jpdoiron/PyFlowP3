from pyrr import Matrix33

from PyFlow.Core.Settings import Colors
from ..Core.AGraphCommon import *
from ..Core.Pin import PinWidgetBase


class Matrix33Pin(PinWidgetBase):
    """doc string for Matrix33Pin"""
    def __init__(self, name, parent, dataType, direction, **kwargs):
        super(Matrix33Pin, self).__init__(name, parent, dataType, direction, **kwargs)
        self.setDefaultValue(Matrix33())

    def supportedDataTypes(self):
        return (DataTypes.Matrix33,)

    @staticmethod
    def color():
        return Colors.Matrix33

    @staticmethod
    def pinDataTypeHint():
        return DataTypes.Matrix33, Matrix33()

    def serialize(self):
        data = PinWidgetBase.serialize(self)
        m = self.currentData()
        data['value'] = [m.c1.tolist(), m.c2.tolist(), m.c3.tolist()]
        return data

    def setData(self, data):
        if isinstance(data, Matrix33):
            self._data = data
        elif isinstance(data, list) and len(data) == 3:
            self._data = Matrix33([data[0], data[1], data[2]])
        else:
            self._data = self.defaultValue()
        PinWidgetBase.setData(self, self._data)
