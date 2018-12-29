from tensorflow import Tensor

from PyFlow.Core.AGraphCommon import DataTypes
from PyFlow.Core.Pin import PinWidgetBase
from PyFlow.Core.Settings import Colors


class LayerPin(PinWidgetBase):
    '''doc string for MyPins'''
    def __init__(self, name, parent, dataType, direction, **kwargs):
        super(LayerPin, self).__init__(name, parent, dataType, direction, **kwargs)
        self.setDefaultValue(None)

    def supportedDataTypes(self):
        return (DataTypes.Layer,)

    def serialize(self):
        data = PinWidgetBase.serialize(self)
        data['value'] = None
        return data

    @staticmethod
    def color():
        return Colors.Layer

    @staticmethod
    def pinDataTypeHint():
        return DataTypes.Layer, None

    def setData(self, data):
        if isinstance(data, Tensor):
            self._data = data
        else:
            self._data = self.defaultValue()

        PinWidgetBase.setData(self, self._data)

