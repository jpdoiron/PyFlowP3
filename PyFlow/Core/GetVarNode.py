"""@file GetVarNode.py

Builtin node to acess variable value.
"""
from .AbstractGraph import *
from .Settings import *
from . import Node
from PySide2.QtWidgets import QStyle
from PySide2.QtWidgets import QGraphicsItem
from PySide2 import QtCore
from PySide2 import QtGui
from ..Commands import RemoveNodes


## Variable getter node
class GetVarNode(Node, NodeBase):
    def __init__(self, name, graph, var):
        super(GetVarNode, self).__init__(name, graph)
        self.var = var
        self.out = self.addOutputPin('val', self.var.dataType, hideLabel=True)
        self.var.valueChanged.connect(self.onVarValueChanged)
        self.var.nameChanged.connect(self.onVarNameChanged)
        self.var.killed.connect(self.kill)
        self.var.dataTypeChanged.connect(self.onVarDataTypeChanged)
        self.label().hide()
        self.label().opt_font.setPointSizeF(6.5)

    def boundingRect(self):
        return QtCore.QRectF(-5, -3, self.w, 20)

    def serialize(self):
        template = Node.serialize(self)
        template['meta']['var'] = self.var.serialize()
        return template

    def onUpdatePropertyView(self, formLayout):
        self.var.onUpdatePropertyView(formLayout)

    def onVarDataTypeChanged(self, dataType):
        cmd = RemoveNodes([self], self.graph())
        self.graph().undoStack.push(cmd)

    def postCreate(self, template):
        self.label().setPlainText(self.var.name)
        Node.postCreate(self, template)

    def onVarNameChanged(self, newName):
        self.label().setPlainText(newName)
        self.setName(newName)
        self.updateNodeShape(label=self.label().toPlainText())

    def onVarValueChanged(self):
        push(self.out)

    @staticmethod
    def category():
        return 'Variables'

    def paint(self, painter, option, widget):
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtCore.Qt.darkGray)

        color = Colors.NodeBackgrounds.lighter(150)
        if self.isSelected():
            color = color.lighter(150)

        linearGrad = QtGui.QRadialGradient(QtCore.QPointF(40, 40), 300)
        linearGrad.setColorAt(0, color)
        linearGrad.setColorAt(1, color.lighter(180))
        br = QtGui.QBrush(linearGrad)
        painter.setBrush(br)
        pen = QtGui.QPen(QtCore.Qt.black, 0.5)
        if option.state & QStyle.State_Selected:
            pen.setColor(Colors.Yellow)
            pen.setStyle(self.opt_pen_selected_type)
        painter.setPen(pen)
        painter.drawRoundedRect(self.boundingRect(), 7, 7)
        painter.setFont(self.label().opt_font)
        pen.setColor(self.var.widget.color)
        painter.setPen(pen)
        painter.drawText(self.boundingRect(), QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter, self.name)

    def compute(self):
        self.out.setData(self.var.value)
