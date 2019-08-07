from graph_pb2 import Graph
import os
import sys
from graph_pb2 import FeatureNode, FeatureEdge

node_name = {
    FeatureNode.TOKEN:"TOKEN", 
    FeatureNode.AST_ELEMENT:"AST_ELEMENT", 
    FeatureNode.IDENTIFIER_TOKEN:"IDENTIFIER_TOKEN",
    FeatureNode.FAKE_AST:"FAKE_AST",
    FeatureNode.SYMBOL_TYP:"SYMBOL_TYP", 
    FeatureNode.COMMENT_LINE:"COMMENT_LINE",
    FeatureNode.TYPE:"TYPE"
}

edge_name = {
    FeatureEdge.NEXT_TOKEN:"NEXT_TOKEN", 
    FeatureEdge.AST_CHILD:"AST_CHILD", 
    FeatureEdge.LAST_WRITE:"LAST_WRITE",
    FeatureEdge.LAST_USE:"LAST_USE", 
    FeatureEdge.COMPUTED_FROM:"COMPUTED_FROM", 
    FeatureEdge.RETURNS_TO:"RETURNS_TO",
    FeatureEdge.FORMAL_ARG_NAME:"FORMAL_ARG_NAME", 
    FeatureEdge.GUARDED_BY:"GUARDED_BY", 
    FeatureEdge.GUARDED_BY_NEGATION:"GUARDED_BY_NEGATION",
    FeatureEdge.LAST_LEXICAL_USE:"LAST_LEXICAL_USE",
    FeatureEdge.ASSIGNABLE_TO:"ASSIGNABLE_TO", 
    FeatureEdge.ASSOCIATED_TOKEN:"ASSOCIATED_TOKEN",
    FeatureEdge.HAS_TYPE:"HAS_TYPE", 
    FeatureEdge.ASSOCIATED_SYMBOL:"ASSOCIATED_SYMBOL"
}

def get_info(graph):

    for node in graph.node:

        print("node id:%s, node contents:%s, node type:%s" % (node.id, node.contents, node_name.setdefault(node.type, "Unknown")))

    for edge in graph.edge:

        print("edge source:%s, edge destination:%s, edge type:%s" % (edge.sourceId, edge.destinationId, edge_name.setdefault(edge.type, "Unknown")))


def open_proto(filename):

    print(filename)

    with open(filename, "rb") as f:

        g = Graph()
        g.ParseFromString(f.read())

        get_info(g)


file = sys.argv[1]

open_proto(file)