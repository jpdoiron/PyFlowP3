import pickle

import matplotlib.pyplot as plt

from ..Core import Node
from ..Core.AbstractGraph import *


class show_history(Node):
    def __init__(self, name, graph):
        super(show_history, self).__init__(name, graph)
        self.in0_pin = self.addInputPin('in', DataTypes.Exec, self.compute)
        self.history_pin = self.addInputPin('history', DataTypes.Any)
        #self.out0 = self.addOutputPin('out0', DataTypes.Bool)
        #pinAffects(self.inp0, self.out0)

    @staticmethod
    def pinTypeHints():
        '''
            used by nodebox to suggest supported pins
            when drop wire from pin into empty space
        '''
        return {'inputs': [DataTypes.Exec], 'outputs': []}

    @staticmethod
    def category():
        '''
            used by nodebox to place in tree
            to make nested one - use '|' like this ( 'CatName|SubCatName' )
        '''
        return 'Common'

    @staticmethod
    def keywords():
        '''
            used by nodebox filter while typing
        '''
        return []

    @staticmethod
    def description():
        '''
            used by property view and node box widgets
        '''
        return 'default description'

    def compute(self):
        '''
            1) get data from inputs
            2) do stuff
            3) put data to outputs
            4) call output execs
        '''
        try:

            import codecs
            serialized = self.history_pin.getData()

            history = pickle.loads(codecs.decode(serialized.encode(), "base64"))

            plt.plot(history['acc'])
            plt.plot(history['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()

            #self.out0.setData(str_data.upper())
        except Exception as e:
            import traceback
            import sys
            traceback.print_exception(type(e), e, sys.exc_info()[2], limit=1, file=sys.stdout)

