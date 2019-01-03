import numpy as np
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split

from ..Core import Node
from ..Core.AbstractGraph import *


class DataSet(Node):
    def __init__(self, name, graph):
        super(DataSet, self).__init__(name, graph)
        self.filename_pin = self.addInputPin('Filename', DataTypes.Files, defaultValue="")
        self.split_pin = self.addInputPin('Training %', DataTypes.Float,defaultValue=0.7)
        self.training_pin = self.addOutputPin('Training', DataTypes.Array)
        self.testing_pin = self.addOutputPin('Testing', DataTypes.Array)

        self.filename = ""
        self.split=0
        self.lbl_train, self.lbl_valid = np.array([]),np.array([])
        self.training_pin.setData(self.lbl_train.tolist())
        self.training_pin.setData(self.lbl_train.tolist())

        self.InitialInit = False
        #pinAffects(self.inp0, self.out0)

    @staticmethod
    def pinTypeHints():
        '''
            used by nodebox to suggest supported pins
            when drop wire from pin into empty space
        '''
        return {'inputs': [DataTypes.Bool], 'outputs': [DataTypes.Bool]}

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

    def ReloadData(self):

        if self.filename == self.filename_pin.getData() and  self.InitialInit:
            return

        self.filename = self.filename_pin.getData()

        if self.filename == "":
            return

        self.split = self.split_pin.getData()
        print("reload", self.filename)

        try:
            if self.filename!="":
                with open(self.filename) as f:
                    lines = f.readlines()

                self.lbl_train, self.lbl_valid = train_test_split(np.array(lines), train_size=self.split)


        except Exception as e:
            print("{}\nError loading file: {}".format(self.filename, e))

    @staticmethod
    def description():
        '''
            used by property view and node box widgets
        '''
        return 'default description'

    def postCreate(self, jsonTemplate):
        Node.postCreate(self, jsonTemplate)

        print("post",self.filename )

    def compute(self):


        '''
            1) get data from inputs
            2) do stuff
            3) put data to outputs
            4) call output execs
        '''
        try:
            self.ReloadData()

            if self.filename != self.filename_pin.getData() or not self.InitialInit:
                self.ReloadData()
                self.training_pin.setData(self.lbl_train.tolist())
                self.testing_pin.setData(self.lbl_valid.tolist())
                self.InitialInit = True

        except Exception as e:
            print("{}\nError setting data: {}".format(self.filename,e))


