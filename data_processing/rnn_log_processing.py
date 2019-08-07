from graph_pb2 import FeatureNode, FeatureEdge
from collections import defaultdict
import numpy as np
from dpu_utils.codeutils import split_identifier_into_parts
from data_processing.graph_features import  get_used_edges_type, get_used_nodes_type

def get_log_samples(graph, seq_length, pad_token, vocabulary, rnn_len):

    node_table = {}
    edge_table = defaultdict(list)
    token_pointer = 0
    token_table = []
    sample_contents = []
    
    semi_node_ids = []
    
    for node in graph.node:

        node_table[node.id] = node
        if (node.type in [FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN]) and token_pointer == 0:
            token_pointer = node.id
            token_table.append(vocabulary.get_id_or_unk_multiple(split_identifier_into_parts(node.contents), seq_length, pad_token))
            
    for edge in graph.edge:
        
        edge_table[edge.sourceId].append(edge)
            
    while(True):
        term_flag = True
        if (len(edge_table[token_pointer]) > 0):
            for edge in edge_table[token_pointer]:
                if edge.type == FeatureEdge.NEXT_TOKEN:
                    term_flag = False
#                    id_in_order.append(token_pointer)
                    if node_table[token_pointer].type == FeatureNode.TOKEN and node_table[token_pointer].contents == "SEMI":
                        semi_node_ids.append(len(token_table))
                    token_pointer = edge.destinationId
                    token_table.append(vocabulary.get_id_or_unk_multiple(split_identifier_into_parts(node_table[token_pointer].contents), seq_length, pad_token))
                    break
            if term_flag:
                break
        else:
#            print("warning: unable to find next node")
            break
            
    for semi_node_id in semi_node_ids:
        
        if semi_node_id < rnn_len:
            sample_content = [vocabulary.get_id_or_unk_multiple(split_identifier_into_parts(" "), seq_length, pad_token)] * rnn_len
            sample_content[-semi_node_id:] = token_table[:semi_node_id]
        else:
            sample_content = token_table[semi_node_id - rnn_len : semi_node_id]
            
        sample_contents.append(np.array([sample_content]))
        
    return sample_contents
        
    