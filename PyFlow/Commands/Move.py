from PySide2.QtWidgets import QUndoCommand


class Move(QUndoCommand):
    '''
    Moves nodes
    '''
    def __init__(self, nodesInfo, graph):
        super(Move, self).__init__()
        self.graph = graph
        self.nodesInfo = nodesInfo
        self.setText("Move")

    def undo(self):
        for nodeUid in self.nodesInfo:
            self.graph.nodes[nodeUid].setPos(self.nodesInfo[nodeUid]['from'])

    def redo(self):
        for nodeUid in self.nodesInfo:
            self.graph.nodes[nodeUid].setPos(self.nodesInfo[nodeUid]['to'])
