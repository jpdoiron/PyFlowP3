from ..Core.Pin import PinWidgetBase
from ..Core.AGraphCommon import *


class StringPin(PinWidgetBase):
    """doc string for StringPin"""
    def __init__(self, name, parent, dataType, direction, **kwargs):
        super(StringPin, self).__init__(name, parent, dataType, direction, **kwargs)
        self.setDefaultValue("")

    def supportedDataTypes(self):
        return (DataTypes.String,)

    @staticmethod
    def color():
        return Colors.String

    @staticmethod
    def pinDataTypeHint():
        return DataTypes.String, ''

    def setData(self, data):
        try:
            self._data = str(data)
        except:
            self._data = self.defaultValue()
        PinWidgetBase.setData(self, self._data)
