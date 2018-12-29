from PySide2.QtWidgets import QUndoCommand
from ..Core import Edge
from uuid import UUID


class RemoveEdges(QUndoCommand):
    '''
    Disconnects pins
    '''
    def __init__(self, graph, jsonTemplates):
        super(RemoveEdges, self).__init__()
        self.setText('Remove edges')
        self.jsonTemplates = jsonTemplates
        self.graph = graph

    def undo(self):
        for edgeJson in self.jsonTemplates:
            Edge.deserialize(edgeJson, self.graph)

    def redo(self):
        for edgeJson in self.jsonTemplates:
            uid = UUID(edgeJson['uuid'])
            if uid in self.graph.edges:
                self.graph.removeEdge(self.graph.edges[uid])
