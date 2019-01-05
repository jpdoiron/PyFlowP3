from PyFlow.Core.Settings import Colors
from ..Core.AGraphCommon import *
from ..Core.Pin import PinWidgetBase


class Files(PinWidgetBase):
    '''doc string for Files'''
    def __init__(self, name, parent, dataType, direction, **kwargs):
        super(Files, self).__init__(name, parent, dataType, direction, **kwargs)
        self.setDefaultValue("")

    def supportedDataTypes(self):
        return (DataTypes.Files,)

    @staticmethod
    def color():
        return Colors.Bool


    @staticmethod
    def pinDataTypeHint():
        return DataTypes.Files, ""

    def setData(self, data):
        try:
            self._data = str(data)
        except:
            self._data = self.defaultValue()

        PinWidgetBase.setData(self, self._data)
