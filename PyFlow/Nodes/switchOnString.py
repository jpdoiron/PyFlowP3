from PySide2.QtWidgets import QInputDialog
from PySide2.QtWidgets import QMenu

from ..Core import Node
from ..Core.AbstractGraph import *
from ..Core.Pin import PinWidgetBase


class switchOnString(Node):
    def __init__(self, name, graph):
        super(switchOnString, self).__init__(name, graph)
        self.inExecPin = self.addInputPin('inExec', DataTypes.Exec, self.compute, hideLabel=True)
        self.defaultPin = None
        self.outString = None
        self.inString = self.addInputPin('String', DataTypes.String)
        self.menu = QMenu()
        self.action = self.menu.addAction('add pin')
        self.action.triggered.connect(self.addOutPin)
        self.actionDebug = self.menu.addAction('debug')
        self.actionDebug.triggered.connect(self.OnDebug)
        self._map = {}

    def renameOutPin(self, oldName, newName):
        if oldName in self._map:
            self._map[oldName].setName(newName)

    def OnDebug(self):
        print((list(self._map.keys())))

    def addOutPin(self):
        name = self.getUniqPinName("option")
        p = self.addOutputPin(name, DataTypes.Exec)
        renameAction = p.menu.addAction("rename")
        killAction = p.menu.addAction("kill")

        def OnKill():
            self._map.pop(p.name)
            p.kill()
        killAction.triggered.connect(OnKill)

        def OnRename():
            res = QInputDialog.getText(None, 'Rename pin', 'label')
            if res[1]:
                newName = self.getUniqPinName(res[0])
                self._map[newName] = self._map.pop(p.name)
                p.setName(newName)
        renameAction.triggered.connect(OnRename)
        pinAffects(self.inExecPin, p)
        self._map[name] = p

    def contextMenuEvent(self, event):
        self.menu.exec_(event.screenPos())

    @staticmethod
    def pinTypeHints():
        return {'inputs': [DataTypes.Exec, DataTypes.String], 'outputs': [DataTypes.Exec]}

    @staticmethod
    def category():
        return 'FlowControl'

    @staticmethod
    def keywords():
        return []

    @staticmethod
    def description():
        return 'Execute output depending on input string'

    def postCreate(self, jsonTemplate):
        Node.postCreate(self, jsonTemplate)

        # restore dynamically created  outputs
        if len(jsonTemplate['outputs']) == 0:
            self.defaultPin = self.addOutputPin('Default', DataTypes.Exec)
            self.outString = self.addOutputPin('stringOut', DataTypes.String, hideLabel=True)
            self.addOutPin()
            self.addOutPin()
        else:
            for out in jsonTemplate['outputs']:
                PinWidgetBase.deserialize(self, out)

    @threaded
    def compute(self):
        string = self.inString.getData()
        self.outString.setData(string)
        if string in self._map:
            self._map[string].call()
        else:
            self.defaultPin.call()
