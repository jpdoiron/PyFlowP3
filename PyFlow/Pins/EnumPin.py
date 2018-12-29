from PyFlow.Core.Settings import Colors
from ..Core.AGraphCommon import *
from ..Core.Enums import ENone
from ..Core.Pin import PinWidgetBase


class EnumPin(PinWidgetBase):
    '''doc string for EnumPin'''
    def __init__(self, name, parent, dataType, direction, **kwargs):
        super(EnumPin, self).__init__(name, parent, dataType, direction, **kwargs)
        self._userStruct = kwargs['userStructClass']
        self.setDefaultValue(self._userStruct(0))

    def serialize(self):
        dt = super(PinWidgetBase, self).serialize()
        dt['value'] = int(dt['value'])
        return dt

    def supportedDataTypes(self):
        return (DataTypes.Int, DataTypes.Enum,)

    @staticmethod
    def color():
        return Colors.Gray

    @staticmethod
    def pinDataTypeHint():
        return DataTypes.Enum, ENone

    def setData(self, data):
        if isinstance(data, self._userStruct) and data != self._userStruct:
            self._data = data
        if isinstance(data, int):
            try:
                self._data = self._userStruct(data)
            except:
                self._data = self.defaultValue()
        PinWidgetBase.setData(self, self._data)
