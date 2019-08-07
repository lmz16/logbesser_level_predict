from graph_pb2 import FeatureNode, FeatureEdge
from collections import defaultdict
import numpy as np
from dpu_utils.codeutils import split_identifier_into_parts
from data_processing.graph_features import  get_used_edges_type, get_used_nodes_type

def get_log_samples(graph, max_path_len, node_seq_length, pad_token, vocabulary):

    successor_table = defaultdict(set)
    predecessor_table = defaultdict(set)
    edge_table = defaultdict(list)
    node_table = {}
    semi_node_ids = []
    samples = []
    token_pointer = 0
#    id_in_order = []

    for node in graph.node:

        node_table[node.id] = node
        if (node.type in [FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN]) and token_pointer == 0:
            token_pointer = node.id
#            id_in_order.append(node.id)
            
    for edge in graph.edge:
        successor_table[edge.sourceId].add(edge.destinationId)
        predecessor_table[edge.destinationId].add(edge.sourceId)
        edge_table[edge.sourceId].append(edge)
        
#    while(True):
#        term_flag = True
#        if len(edge_table[token_pointer]) > 0:
#            print(node_table[token_pointer].contents)
#            for edge in edge_table[token_pointer]:
#                if (edge.type == FeatureEdge.NEXT_TOKEN) and (edge.sourceId != token_pointer):
#                    token_pointer = edge.sourceId
#                    term_flag = False
#                    break
#            if term_flag:
#                break
#        else:
#            print("warning: single node")
#            break
            

    while(True):
        term_flag = True
        if (len(edge_table[token_pointer]) > 0):
            for edge in edge_table[token_pointer]:
                if edge.type == FeatureEdge.NEXT_TOKEN:
                    term_flag = False
#                    id_in_order.append(token_pointer)
                    if node_table[token_pointer].type == FeatureNode.TOKEN and node_table[token_pointer].contents == "SEMI":
                        semi_node_ids.append(token_pointer)
                    token_pointer = edge.destinationId
                    break
            if term_flag:
                break
        else:
#            print("warning: unable to find next node")
            break
    
    for semi_node_id in semi_node_ids:
        reachable_node_ids = []
        successor_ids = [semi_node_id]
        predecessor_ids = [semi_node_id]
        
#        next_token_node_id = None
#        last_token_node_id = None
        
#        temp_num = id_in_order.index(semi_node_id)
#        if temp_num == 0:
#            last_token_node_id = semi_node_id
#            next_token_node_id = id_in_order[temp_num + 1]
#        elif temp_num == len(id_in_order) - 1:
#            next_token_node_id = semi_node_id
#            last_token_node_id = id_in_order[temp_num - 1]
#        else:
#            next_token_node_id = id_in_order[temp_num + 1]
#            last_token_node_id = id_in_order[temp_num - 1]
            

        for _ in range(max_path_len):
            reachable_node_ids += successor_ids
            reachable_node_ids += predecessor_ids
            successor_ids = list(set([elem for n_id in successor_ids for elem in successor_table[n_id]]))
            predecessor_ids = list(set([elem for n_id in predecessor_ids for elem in predecessor_table[n_id]]))
        
        reachable_node_ids += successor_ids
        reachable_node_ids += predecessor_ids
        reachable_node_ids = set(reachable_node_ids)
        
#        if next_token_node_id in reachable_node_ids and next_token_node_id != semi_node_id:
#            reachable_node_ids.remove(next_token_node_id)
#        if last_token_node_id in reachable_node_ids and last_token_node_id != semi_node_id:
#            reachable_node_ids.remove(last_token_node_id)

        sub_nodes = [node_table[node_id] for node_id in reachable_node_ids]

        sub_edges =  [edge for node in sub_nodes for edge in edge_table[node.id]
                      if edge.sourceId in reachable_node_ids and edge.destinationId in reachable_node_ids]

        sub_graph = (sub_nodes, sub_edges)

        sample_data = compute_sample_data(sub_graph, [semi_node_id], node_seq_length, pad_token, vocabulary)

        samples.append(sample_data)

    return samples

    
def compute_sample_data(sub_graph, identifier_token_node_ids, seq_length, pad_token, vocabulary):

    used_node_types = get_used_nodes_type()
    used_edge_types = get_used_edges_type()

    node_representations = []
    id_to_index_map = {}
    ind = 0

    (sub_nodes, sub_edges) = sub_graph

    for node in sub_nodes:
        if node.type in used_node_types:
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
        if edge.type in used_edge_types and edge.sourceId in id_to_index_map and edge.destinationId in id_to_index_map:

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