from graph_pb2 import FeatureNode, FeatureEdge
from collections import defaultdict
import numpy as np
from dpu_utils.codeutils import split_identifier_into_parts
from data_processing.graph_features import  get_used_edges_type, get_used_nodes_type

'''
Used to create input samples from a given graph 

:sub_graph: input graph
:identifier_token_node_ids: usage/declaration node ids
:seq_length: length of node representation
:pad_token: vocabulary pad token
:slot_token: vocabulary slot token
:vocabulary: corpus token vocabulary
:exception_node_ids: when computing method usage information, there is a chance that a method declaration node is reachable 
from one of the usage nodes. This node should also be masked with a <SLOT> token, but should not be used in consequent decoding steps
(because it is not a usage node, but a declaration node), thus it is marked as an exception
'''
def compute_sample_data(sub_graph, identifier_token_node_ids, seq_length, pad_token, slot_token, vocabulary, exception_node_ids = []):

    used_node_types = get_used_nodes_type()
    used_edge_types = get_used_edges_type()

    node_representations = []
    id_to_index_map = {}
    ind = 0

    (sub_nodes, sub_edges) = sub_graph

    for node in sub_nodes:
        if node.type in used_node_types:
            if node.id in exception_node_ids:
                node_representation = [pad_token for _ in range(seq_length)]
                node_representation[0] = slot_token
            else:
                node_representation = vocabulary.get_id_or_unk_multiple(split_identifier_into_parts(node.contents), seq_length, pad_token)

            node_representations.append(node_representation)
            id_to_index_map[node.id] = ind
            ind += 1

    n_nodes = len(node_representations)
    n_types = len(used_edge_types)
    node_representations = np.array(node_representations)
    num_incoming_edges_per_type = np.zeros((n_nodes, n_types))
    num_outgoing_edges_per_type = np.zeros((n_nodes, n_types))
    adj_lists = defaultdict(list)

    for edge in sub_edges:
        if edge.type in used_edge_types \
                and edge.sourceId in id_to_index_map \
                and edge.destinationId in id_to_index_map:

            type_id = used_edge_types.index(edge.type)
            adj_lists[type_id].append([id_to_index_map[edge.sourceId], id_to_index_map[edge.destinationId]])
            num_incoming_edges_per_type[id_to_index_map[edge.destinationId], type_id] += 1
            num_outgoing_edges_per_type[id_to_index_map[edge.sourceId], type_id] += 1

    final_adj_lists = {edge_type: np.array(sorted(adj_list), dtype=np.int32)
                       for edge_type, adj_list in adj_lists.items()}

    # Add empty entries for types with no adjacency lists
    for i in range(len(used_edge_types)):
        if i not in final_adj_lists:
            final_adj_lists[i] = np.zeros((0, 2), dtype=np.int32)


    identifier_nodes = [id_to_index_map[node_id] for node_id in identifier_token_node_ids]

    return (identifier_nodes, node_representations, final_adj_lists, \
           num_incoming_edges_per_type, num_outgoing_edges_per_type)


'''
Extract log information from a given graph

:graph: input graph sample
:max_path_len: number of GGNN timesteps (used to remove nodes/edges unreachable in this amount of timesteps from <SLOT> nodes)
:node_rep_len: maximum number of subtokens in node representation
:pad_token: vocabulary pad token
:slot_token: vocabulary slot token
:vocabulary: corpus token vocabulary
'''
def get_log_samples(graph, max_path_len, node_seq_length, pad_token, slot_token, vocabulary):

    successor_table = defaultdict(set)
    predecessor_table = defaultdict(set)
    edge_table = defaultdict(list)
    node_table = {}
    semi_node_ids = []
    samples = []
    token_pointer = 0

    for node in graph.node:

        node_table[node.id] = node
        if (node.type in [FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN]) and token_pointer == 0:
            token_pointer = node.id
            
    for edge in graph.edge:
        successor_table[edge.sourceId].add(edge.destinationId)
        predecessor_table[edge.destinationId].add(edge.sourceId)
        edge_table[edge.sourceId].append(edge)
        
    while(True):
        term_flag = True
        if len(edge_table[token_pointer]) > 0:
            print(node_table[token_pointer].contents)
            for edge in edge_table[token_pointer]:
                if (edge.type == FeatureEdge.NEXT_TOKEN) and (edge.sourceId != token_pointer):
                    token_pointer = edge.sourceId
                    term_flag = False
                    break
            if term_flag:
                break
        else:
            print("warning: single node")
            break
            

    while(True):
        term_flag = True
        if (len(edge_table[token_pointer]) > 0):
            for edge in edge_table[token_pointer]:
                if (edge.type == FeatureEdge.NEXT_TOKEN) and (edge.destinationId != token_pointer):
                    term_flag = False
                    if node_table[token_pointer].type == FeatureNode.TOKEN and node_table[token_pointer].contents == "SEMI":
                        semi_node_ids.append(token_pointer)
                    token_pointer = edge.destinationId
                    break
            if term_flag:
                break
        else:
            print("error: unable to find next node")
            break
            
    print(len(semi_node_ids))
    
    for semi_node_id in semi_node_ids:
        reachable_node_ids = []
        successor_ids = [semi_node_id]
        predecessor_ids = [semi_node_id]

        for _ in range(max_path_len):
            reachable_node_ids += successor_ids
            reachable_node_ids += predecessor_ids
            successor_ids = list(set([elem for n_id in successor_ids for elem in successor_table[n_id]]))
            predecessor_ids = list(set([elem for n_id in predecessor_ids for elem in predecessor_table[n_id]]))
        
        reachable_node_ids += successor_ids
        reachable_node_ids += predecessor_ids
        reachable_node_ids = set(reachable_node_ids)

        sub_nodes = [node_table[node_id] for node_id in reachable_node_ids]

        sub_edges =  [edge for node in sub_nodes for edge in edge_table[node.id]
                      if edge.sourceId in reachable_node_ids and edge.destinationId in reachable_node_ids]

        sub_graph = (sub_nodes, sub_edges)

        sample_data = compute_sample_data(sub_graph, [semi_node_id], node_seq_length, pad_token, slot_token, vocabulary)

        samples.append(sample_data)

    return samples

def debug_graph_infomation(graph):
    for node in graph.node:
        print("node.id:%s   node.type:%s    node.content:%s" % (node.id, node.type, node.content))
        
def get_log_content_samples(graph, max_path_len, node_seq_length, pad_token, slot_token, vocabulary):
    
    successor_table = defaultdict(set)
    predecessor_table = defaultdict(set)
    edge_table = defaultdict(list)
    node_table = {}
    var_node_ids = []
    samples = []
    
    for node in graph.node:

        node_table[node.id] = node
        
        if node.type == FeatureNode.SYMBOL_VAR:
            var_node_ids.append(node.id)
            
    for edge in graph.edge:
        successor_table[edge.sourceId].add(edge.destinationId)
        predecessor_table[edge.destinationId].add(edge.sourceId)
        edge_table[edge.sourceId].append(edge)
        
    