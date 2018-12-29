"""@file CommentNode.py
"""
from ..Core.AbstractGraph import *
from ..Core.Settings import *
from ..Core import Node
from ..Core import NodeName
from types import MethodType
from PySide2.QtWidgets import QGraphicsTextItem
from PySide2.QtWidgets import QGraphicsItem
from PySide2.QtWidgets import QGraphicsItemGroup
from PySide2.QtWidgets import QStyle
from PySide2.QtWidgets import QLabel
from PySide2.QtWidgets import QLineEdit
from PySide2.QtWidgets import QTextBrowser
from PySide2.QtWidgets import QPushButton
from PySide2.QtWidgets import QMenu
from PySide2.QtWidgets import QColorDialog
from PySide2 import QtGui
from PySide2 import QtCore


class commentNodeName(NodeName):
    """doc string for commentNodeName"""
    def __init__(self, parent, bUseTextureBg=False, color=Colors.AbsoluteBlack):
        super(commentNodeName, self).__init__(parent, bUseTextureBg, color=color)
        self.color = color
        self.color.setAlpha(80)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setDefaultTextColor(QtGui.QColor(255, 255, 255, 255))
        self.roundCornerFactor = 1.0
        self.setTextInteractionFlags(QtCore.Qt.TextEditorInteraction)
        self.width = self.document().documentLayout().documentSize().width()

    def keyPressEvent(self, event):

        QGraphicsTextItem.keyPressEvent(self, event)
        modifires = event.modifiers()
        key = event.key()

        if key == QtCore.Qt.Key_Escape:
            # clear selection
            cursor = QtGui.QTextCursor(self.document())
            cursor.clearSelection()
            self.setTextCursor(cursor)
            # finish editing
            self.setEnabled(False)
            self.setEnabled(True)
        elif not key == QtCore.Qt.Key_Backspace:
            # if backspace is pressed do not change width
            width = self.document().documentLayout().documentSize().width()
            self.h = self.document().documentLayout().documentSize().height()
            # change width if needed
            if width >= self.parentItem().rect.width():
                self.width = width
                self.adjustSizes()

        self.width = self.parentItem().rect.width()
        self.setTextWidth(self.width)
        self.update()

    def adjustSizes(self):
        self.parentItem().rect.setRight(self.width)
        self.setTextWidth(self.width)
        self.h = self.document().documentLayout().documentSize().height()
        self.update()
        self.parentItem().update()

    def paint(self, painter, option, widget):
        QGraphicsTextItem.paint(self, painter, option, widget)
        r = QtCore.QRectF(option.rect)
        r.setWidth(self.width)
        r.setX(0.25)
        r.setY(0.25)
        b = QtGui.QLinearGradient(0, 0, 0, r.height())
        b.setColorAt(0, QtGui.QColor(0, 0, 0, 0))
        b.setColorAt(0.25, self.color)
        b.setColorAt(1, self.color)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(b)
        painter.drawRoundedRect(r, self.roundCornerFactor, self.roundCornerFactor)

    def hoverEnterEvent(self, event):
        NodeName.hoverEnterEvent(self, event)


## Comment node
#
# Can drag intersected nodes.
# You can also specify color and resize it.
class commentNode(Node, NodeBase):
    def __init__(self, name, graph, bUseTextureBg=False, headColor=Colors.AbsoluteBlack):
        super(commentNode, self).__init__(name, graph, headColor=headColor)
        self.color = Colors.AbsoluteBlack
        self.color.setAlpha(80)
        self.nodesToMove = {}
        self.lastNodePos = self.scenePos()
        self.rect = self.childrenBoundingRect()
        self.initialRectWidth = 0.0
        self.initialRectHeight = 0.0
        self.mousePressPos = self.scenePos()
        self.resizeDirection = (0, 0)
        self.bResize = False
        self.bMove = False
        self.minWidth = 100.0
        self.minHeight = 100.0
        self.lastMousePos = QtCore.QPointF()
        self.setZValue(-2)

    def onChangeColor(self):
        res = QColorDialog.getColor(self.color, None, 'Comment node color setup')
        if res.isValid():
            res.setAlpha(80)
            self.color = res
            self.label().color = res
            self.update()
            self.label().update()

    def serialize(self):
        template = Node.serialize(self)
        template['meta']['commentNode'] = {
            'w': self.rect.right(),
            'h': self.rect.bottom(),
            'labelHeight': self.label().h,
            'text': self.label().toPlainText(),
            'color': (self.color.toTuple())
        }
        return template

    def postCreate(self, jsonTemplate):
        Node.postCreate(self, jsonTemplate)
        # restore text and size
        width = self.minWidth
        height = self.minHeight
        labelHeight = self.label().h
        text = self.__class__.__name__
        # initial color is black
        color = self.color
        try:
            # if copied in runtime
            width = jsonTemplate['meta']['commentNode']['w']
            height = jsonTemplate['meta']['commentNode']['h']
            labelHeight = jsonTemplate['meta']['commentNode']['labelHeight']
            text = jsonTemplate['meta']['commentNode']['text']
            color = QtGui.QColor(*jsonTemplate['meta']['commentNode']['color'])
        except:
            pass

        self.rect.setRight(width)
        self.rect.setBottom(height)
        self.color = color
        self.update()
        self.scene().removeItem(self.label())
        delattr(self, 'label')
        self.label = weakref.ref(commentNodeName(self, False, Colors.White))
        self.label().setPlainText(text)
        self.label().width = self.rect.width()
        self.label().h = labelHeight
        self.label().color = color
        self.label().update()
        self.label().adjustSizes()

    @staticmethod
    def isInRange(mid, val, width=10):
        '''check if val inside strip'''

        leftEdge = mid - width
        rightEdge = mid + width
        return leftEdge <= val <= rightEdge

    def boundingRect(self):
        return self.rect

    def mousePressEvent(self, event):
        QGraphicsItem.mousePressEvent(self, event)
        self.mousePressPos = event.scenePos()

        pBottomRight = self.rect.bottomRight()
        bottomRightRect = QtCore.QRectF(pBottomRight.x() - 6, pBottomRight.y() - 6, 5, 5)

        # detect where on the node
        if bottomRightRect.contains(event.pos()):
            self.initialRectWidth = self.rect.width()
            self.initialRectHeight = self.rect.height()
            self.resizeDirection = (1, 1)
            self.setFlag(QGraphicsItem.ItemIsMovable, False)
            self.bResize = True
        elif event.pos().x() > (self.rect.width() - 20):
            self.initialRectWidth = self.rect.width()
            self.resizeDirection = (1, 0)
            self.setFlag(QGraphicsItem.ItemIsMovable, False)
            self.bResize = True
        elif (event.pos().y() + self.label().defaultHeight) > (self.rect.height() - 30):
            self.initialRectHeight = self.rect.height()
            self.resizeDirection = (0, 1)
            self.setFlag(QGraphicsItem.ItemIsMovable, False)
            self.bResize = True

        self.nodesToMove.clear()
        for node in [i for i in self.collidingItems() if isinstance(i, Node) and not isinstance(i, commentNode)]:
            self.nodesToMove[node] = node.scenePos()

    def mouseMoveEvent(self, event):
        QGraphicsItem.mouseMoveEvent(self, event)
        delta = self.lastMousePos - event.pos()
        # resize
        if self.bResize:
            delta = event.scenePos() - self.mousePressPos
            if self.resizeDirection == (1, 0):
                # right edge resize
                newWidth = delta.x() + self.initialRectWidth
                newWidth = roundup(newWidth, self.graph().grid_size)
                if newWidth > self.minWidth:
                    self.label().width = newWidth
                    self.rect.setRight(newWidth)
                    self.label().adjustSizes()

            elif self.resizeDirection == (0, 1):
                newHeight = delta.y() + self.initialRectHeight
                newHeight = max(roundup(newHeight, self.graph().grid_size), self.label().h + 20.0)
                if newHeight > self.minHeight:
                    # bottom edge resize
                    self.rect.setHeight(newHeight)
            elif self.resizeDirection == (1, 1):
                newWidth = delta.x() + self.initialRectWidth
                newWidth = roundup(newWidth, self.graph().grid_size)

                newHeight = delta.y() + self.initialRectHeight
                newHeight = max(roundup(newHeight, self.graph().grid_size), self.label().h + 20.0)
                if newHeight > self.minHeight and newWidth > self.minWidth:
                    self.label().width = newWidth
                    self.rect.setRight(newWidth)
                    self.label().setTextWidth(newWidth)
                    self.rect.setHeight(newHeight)

            self.update()
            self.label().update()
        self.lastMousePos = event.pos()

    def itemChange(self, change, value):
        if change == self.ItemPositionChange:
            # grid snapping
            value.setX(roundup(value.x() - self.graph().grid_size + self.graph().grid_size / 3.0, self.graph().grid_size))
            value.setY(roundup(value.y() - self.graph().grid_size + self.graph().grid_size / 3.0, self.graph().grid_size))
            value.setY(value.y() - 2)
            try:
                deltaPos = value - self.lastNodePos
                for node, initialPos in self.nodesToMove.items():
                    nodePos = node.scenePos()
                    newPos = nodePos + deltaPos
                    node.setPos(newPos)
                self.lastNodePos = value
            except:
                pass
            return value
        return QGraphicsItem.itemChange(self, change, value)

    @staticmethod
    def getNodesRect(nodes):
        rectangles = []

        if len(nodes) == 0:
            return None

        for n in nodes:
            rectangles.append(n.sceneBoundingRect())

        minx_arr = [i.left() for i in rectangles]
        maxx_arr = [i.right() for i in rectangles]
        miny_arr = [i.top() for i in rectangles]
        maxy_arr = [i.bottom() for i in rectangles]

        min_x = min(minx_arr)
        min_y = min(miny_arr)

        max_x = max(maxx_arr)
        max_y = max(maxy_arr)

        return QtCore.QRect(QtCore.QPoint(min_x, min_y), QtCore.QPoint(max_x, max_y))

    def mouseReleaseEvent(self, event):
        QGraphicsItem.mouseReleaseEvent(self, event)
        self.bResize = False

    def paint(self, painter, option, widget):
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtCore.Qt.darkGray)

        color = Colors.NodeBackgrounds
        if self.isSelected():
            color = color.lighter(150)

        painter.setBrush(self.color)
        pen = QtGui.QPen(QtCore.Qt.black, 0.5)
        if option.state & QStyle.State_Selected:
            pen.setColor(Colors.Yellow)
            pen.setStyle(QtCore.Qt.SolidLine)
        painter.setPen(pen)
        painter.drawRoundedRect(self.rect, self.sizes[4], self.sizes[5])

        pen.setColor(Colors.White)
        painter.setPen(pen)

        # draw bottom right resizer
        pBottomRight = self.rect.bottomRight()
        bottomRightRect = QtCore.QRectF(pBottomRight.x() - 6, pBottomRight.y() - 6, 5, 5)
        painter.drawLine(bottomRightRect.bottomLeft(), bottomRightRect.topRight())

        bottomRightRect.setRight(bottomRightRect.left() + bottomRightRect.width() / 2)
        bottomRightRect.setBottom(bottomRightRect.top() + bottomRightRect.height() / 2)
        painter.drawLine(bottomRightRect.bottomLeft(), bottomRightRect.topRight())

        pen.setWidth(1)
        painter.setPen(pen)

        # draw right resizer
        midY = self.rect.center().y()
        pTop = QtCore.QPoint(self.rect.width() - 5, midY - 5)
        pBottom = QtCore.QPoint(self.rect.width() - 5, midY + 5)
        painter.drawLine(pTop, pBottom)

        # draw bottom resizer
        midX = self.rect.center().x()
        pLeft = QtCore.QPoint(midX - 5, self.rect.bottom() - 5)
        pRight = QtCore.QPoint(midX + 5, self.rect.bottom() - 5)
        painter.drawLine(pLeft, pRight)

    def onUpdatePropertyView(self, formLayout):

        # name
        le_name = QLineEdit(self.getName())
        le_name.setReadOnly(True)
        if self.label().IsRenamable():
            le_name.setReadOnly(False)
            le_name.returnPressed.connect(lambda: self.setName(le_name.text()))
        formLayout.addRow("Name", le_name)

        # uid
        leUid = QLineEdit(str(self.uid))
        leUid.setReadOnly(True)
        formLayout.addRow("Uuid", leUid)

        # type
        leType = QLineEdit(self.__class__.__name__)
        leType.setReadOnly(True)
        formLayout.addRow("Type", leType)

        # pos
        le_pos = QLineEdit("{0} x {1}".format(self.pos().x(), self.pos().y()))
        formLayout.addRow("Pos", le_pos)

        pb = QPushButton("...")
        pb.clicked.connect(self.onChangeColor)
        formLayout.addRow("Color", pb)

        doc_lb = QLabel()
        doc_lb.setStyleSheet("background-color: black;")
        doc_lb.setText("Description")
        formLayout.addRow("", doc_lb)
        doc = QTextBrowser()
        doc.setOpenExternalLinks(True)
        doc.setHtml(self.description())
        formLayout.addRow("", doc)

    @staticmethod
    def category():
        return 'Common'

    @staticmethod
    def keywords():
        return []

    @staticmethod
    def description():
        return 'Can drag intersected nodes. You can also specify color and resize it.'
