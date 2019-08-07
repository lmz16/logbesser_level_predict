import tensorflow as tf
from graph_pb2 import Graph
from dpu_utils.tfmodels import SparseGGNN
from data_processing.sample_inf_processing import SampleMetaInformation, CorpusMetaInformation
import numpy as np
import os
from data_processing import log_graph_processing
from data_processing.graph_features import get_used_edges_type
from random import shuffle
from utils.utils import compute_f1_score
from utils.roc_auc import ROC, AUC, BrierScore
import json
import datetime

class Model:

    def __init__(self, mode, task_id, vocabulary):

        # Initialize parameter values
        self.max_node_seq_len = 32                          # Maximum number of node subtokens
        self.max_var_seq_len = 16                           # Maximum number of variable subtokens
        self.learning_rate = 0.001
        self.ggnn_params = self.get_gnn_params()
        self.vocabulary = vocabulary
        self.voc_size = len(vocabulary)
        self.pad_token_id = self.vocabulary.get_id_or_unk(self.vocabulary.get_pad())
        self.embedding_size1 = self.ggnn_params['hidden_size']
        self.embedding_size2 = 32
        self.embedding_size3 = 16
        self.embedding_size4 = 6
        self.ggnn_dropout = 1.0
        self.label_kind = 6
        self.batch_size = 20000
        self.enable_batching = True
        self.model_type = task_id
        
        if mode == "infer":
            self.enable_batching = False
            

        if mode != 'train' and mode != 'infer':
            raise ValueError("Invalid mode. Please specify \'train\' or \'infer\'...")

        self.graph = tf.Graph()
        self.mode = mode

        with self.graph.as_default():

            self.placeholders = {}
            self.make_model()
            self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto())

            if self.mode == 'train':
                self.make_train_step()
                self.sess.run(tf.global_variables_initializer())


        print ("Model built successfully...")


    def get_gnn_params(self):

        gnn_params = {}
        gnn_params["n_edge_types"] = len(get_used_edges_type())
        gnn_params["hidden_size"] = 64
        gnn_params["edge_features_size"] = {}
        gnn_params["add_backwards_edges"] = True
        gnn_params["message_aggregation_type"] = "sum"
        gnn_params["layer_timesteps"] = [8]
        gnn_params["use_propagation_attention"] = False
        gnn_params["use_edge_bias"] = False
        gnn_params["graph_rnn_activation"] = "relu"
        gnn_params["graph_rnn_cell"] = "gru"
        gnn_params["residual_connections"] = {}
        gnn_params["use_edge_msg_avg_aggregation"] = False

        return gnn_params

    def make_model(self):

        # Create inputs and compute initial node representations
        self.make_inputs()
        self.get_initial_node_representation()

        # Run graph through GGNN layer
        self.gnn_model = SparseGGNN(self.ggnn_params)
        self.gnn_representation = self.gnn_model.sparse_gnn_layer(self.ggnn_dropout,
                                                        self.node_label_representations,
                                                        self.placeholders['adjacency_lists'],
                                                        self.placeholders['num_incoming_edges_per_type'],
                                                        self.placeholders['num_outgoing_edges_per_type'],
                                                        {})

        # Compute average of semi representations
        self.avg_representation = tf.gather(self.gnn_representation, self.placeholders['semi_ids'])
        semi_mask = tf.reshape(self.placeholders['semi_ids_mask'], [-1, self.max_node_seq_len, 1])
        semi_embedding = semi_mask * self.avg_representation
        self.avg_representation = tf.reduce_sum(semi_embedding, axis=1)
        num_semis = tf.reduce_sum(semi_mask, axis=1)
        self.avg_representation /= num_semis

        mid_result1 = tf.layers.dense(inputs=self.avg_representation, units=self.embedding_size1, use_bias=True, bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu)
        mid_result2 = tf.layers.dense(inputs=mid_result1, units=self.embedding_size2, activation=tf.nn.relu)
        mid_result3 = tf.layers.dense(inputs=mid_result2, units=self.embedding_size3, activation=tf.nn.relu)
#        if self.model_type == "c":
#            self.bin_prediction = tf.layers.dense(inputs=mid_result3, units=6)
#            self.prob = tf.nn.softmax(self.bin_prediction)
#        elif self.model_type == "r":
#            mid_result4 = tf.layers.dense(inputs=mid_result3, units=self.embedding_size4, activation=tf.nn.relu)
#            self.float_prediction = tf.layers.dense(inputs=mid_result4, units=1, activation=tf.nn.sigmoid)
#            self.prob = 10.0 * self.float_prediction - 2.5
        self.bin_prediction = tf.layers.dense(inputs=mid_result3, units=6)
        self.prob = tf.nn.softmax(self.bin_prediction)
    
    def make_train_step(self):

#        if self.model_type == "c":
#            #self.train_loss = tf.square(tf.subtract(self.bin_prediction, self.placeholders['targets']))
#            self.train_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.bin_prediction, labels=self.placeholders['targets']))/self.placeholders['num_samples_in_batch']
#        elif self.model_type == "r":
#            self.train_loss = tf.nn.l2_loss(self.prob - tf.to_float(tf.argmax(self.placeholders['targets'], dimension=1)))/self.placeholders['num_samples_in_batch']
        self.train_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.bin_prediction, labels=self.placeholders['targets']))/self.placeholders['num_samples_in_batch']
        self.train_vars = tf.trainable_variables()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        grads_and_vars = self.optimizer.compute_gradients(self.train_loss, var_list=self.train_vars)

        clipped_grads = []

        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, 5.0), var))
            else:
                clipped_grads.append((grad, var))

        self.train_step = self.optimizer.apply_gradients(clipped_grads)


    def make_inputs(self):
        
        # Node token sequences
        self.placeholders['unique_node_labels'] = tf.placeholder(name='unique_labels', shape=[None, self.max_node_seq_len], dtype=tf.int32 )
        self.placeholders['unique_node_labels_mask'] = tf.placeholder(name='unique_node_labels_mask', shape=[None, self.max_node_seq_len], dtype=tf.float32)
        self.placeholders['node_label_indices'] = tf.placeholder(name='node_label_indices', shape=[None], dtype=tf.int32)

        # Graph edge matrices
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2]) for _ in range(self.ggnn_params['n_edge_types'])]
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, self.ggnn_params['n_edge_types']])
        self.placeholders['num_outgoing_edges_per_type'] = tf.placeholder(tf.float32, [None, self.ggnn_params['n_edge_types']])
        
        self.placeholders['semi_ids'] = tf.placeholder(tf.int32, [None, self.max_node_seq_len], name='semi_ids')
        self.placeholders['semi_ids_mask'] = tf.placeholder(tf.float32, [None, self.max_node_seq_len], name='semi_ids_mask')

        self.placeholders['targets'] = tf.placeholder(dtype=tf.float32, shape=[None, self.label_kind], name='target')
        self.placeholders['num_samples_in_batch'] = tf.placeholder(dtype=tf.float32, shape=(1), name='num_samples_in_batch')


    def get_initial_node_representation(self):

        # Compute the embedding of input node sub-tokens
        self.embedding_encoder = tf.get_variable('embedding_encoder', [self.voc_size, self.embedding_size1])
        subtoken_embedding = tf.nn.embedding_lookup(params=self.embedding_encoder, ids=self.placeholders['unique_node_labels'])
        subtoken_ids_mask = tf.reshape(self.placeholders['unique_node_labels_mask'], [-1, self.max_node_seq_len, 1])
        subtoken_embedding = subtoken_ids_mask * subtoken_embedding
        unique_label_representations = tf.reduce_sum(subtoken_embedding, axis=1)
        num_subtokens = tf.reduce_sum(subtoken_ids_mask, axis=1)
        unique_label_representations /= num_subtokens
        self.node_label_representations = tf.gather(params=unique_label_representations,
                                               indices=self.placeholders['node_label_indices'])

    
    def train(self, train_path, val_path, n_epochs, checkpoint_path):

        train_samples, train_labels = self.get_samples(train_path)
        print("Extracted training samples... ", len(train_samples))

        with self.graph.as_default():

            for epoch in range(n_epochs):

                loss = 0
                count = 0

                for graph in train_samples:
                    loss += self.sess.run([self.train_loss, self.train_step], feed_dict=graph)[0]/graph[self.placeholders['num_samples_in_batch']]
                    #print("label: %s, prediction: %s" %(train_labels[count], self.sess.run(self.bin_prediction, feed_dict=graph)))
                    count += 1

                print("Average Epoch Loss:", (loss/len(train_samples)))
                print("Epoch: ", epoch + 1, "/", n_epochs)
                print("---------------------------------------------")


                if (epoch+1) % 5 == 0:

                    saver = tf.train.Saver()
                    saver.save(self.sess, checkpoint_path)

            saver = tf.train.Saver()
            saver.save(self.sess, checkpoint_path)

    
    def get_samples(self, dir_path):

        graph_samples, labels = [], []
        n_files = sum([1 for dirpath, dirs, files in os.walk(dir_path) for filename in files if filename.endswith('proto')])
        n_processed = 0

        for dirpath, dirs, files in os.walk(dir_path):
            for filename in files:
                if filename.endswith('proto'):
                    fname = os.path.join(dirpath, filename)

                    new_samples, new_labels = self.create_samples(fname)

                    if len(new_samples) > 0:
                        graph_samples += new_samples
                        labels += new_labels

                    n_processed += 1
                    print("Processed ", n_processed/n_files * 100, "% of files...")


        zipped = list(zip(graph_samples, labels))
        #shuffle(zipped)
        graph_samples, labels = zip(*zipped)
        
        if self.enable_batching:
            graph_samples, labels = self.make_batch_samples(graph_samples, labels)

        return graph_samples, labels

    
    def create_samples(self, filepath):

        with open(filepath, "rb") as f:

            g = Graph()
            g.ParseFromString(f.read())
            
            true_labels = []

            max_path_len = 8
            
            graph_samples = log_graph_processing.get_log_samples(g, max_path_len, self.max_node_seq_len, self.pad_token_id, self.vocabulary)

            samples, labels, slot_labels = [], [], []
            
            try:
                with open(os.path.splitext(filepath)[0].replace("java", "json"), "r") as f_:
                    labels, slot_labels = json.load(f_)
                    if len(slot_labels) != len(graph_samples) or len(labels) != sum(slot_labels):
                        print("Error: labels and samples don't match, num of labels: %d, num of samples: %d, filename: %s" % (len(labels), len(graph_samples), filepath))
                        os.system("rm -f "+filepath[:-11]+"*")
            except FileNotFoundError as e:
                print("Warning: file not found. It's ok if you are not training the model")
                labels = [self.label_kind - 1] * len(graph_samples)
                slot_labels = [1] * len(graph_samples)

            count = 0

            for i in range(len(graph_samples)):
                if slot_labels[i] != 0:
                    new_sample = self.create_sample(*(graph_samples[i]), labels[count])
                    samples.append(new_sample)
                    true_labels.append(labels[count])
                    count += 1

            return samples, true_labels

    def create_sample(self, semi_id_list, node_representation, adj_lists, incoming_edges, outgoing_edges, label):

        node_rep_mask = (node_representation != self.pad_token_id).astype(int)

        semi_ids = np.zeros((1, self.max_node_seq_len))
        semi_mask = np.zeros((1, self.max_node_seq_len))
        semi_ids[0, 0:len(semi_id_list)] = semi_id_list
        semi_mask[0, 0:len(semi_id_list)] = 1
        
        target = np.zeros((1, self.label_kind))
        target[0, label] = 1

        if self.enable_batching:
            unique_label_subtokens, unique_label_indices = None, None
            unique_label_inverse_indices = node_representation
        else:
            unique_label_subtokens, unique_label_indices, unique_label_inverse_indices = np.unique(node_representation, return_index=True, return_inverse=True, axis=0)

        graph_sample = {
            self.placeholders['unique_node_labels']: unique_label_subtokens,
            self.placeholders['unique_node_labels_mask']: node_rep_mask[unique_label_indices],
            self.placeholders['node_label_indices']: unique_label_inverse_indices,
            self.placeholders['semi_ids']: semi_ids,
            self.placeholders['semi_ids_mask']: semi_mask,
            self.placeholders['num_incoming_edges_per_type']: incoming_edges,
            self.placeholders['num_outgoing_edges_per_type']: outgoing_edges,
            self.placeholders['targets']: target,
            self.placeholders['num_samples_in_batch']: np.ones((1))
        }

        for i in range(self.ggnn_params['n_edge_types']):
            graph_sample[self.placeholders['adjacency_lists'][i]] = adj_lists[i]

        return graph_sample
        
    def infer(self, test_path, checkpoint_path):
    
        test_samples, test_labels = self.get_samples(test_path)
        with self.graph.as_default():

            saver = tf.train.Saver()
            saver.restore(self.sess, checkpoint_path)
            print("Model loaded successfully...")
            
        predictions = []
        softmaxs = []
        pred_index = []
        corrected_points = 0
        baseline_points = 0
        
        cost_matrix = [[0, 0.2, 0.4, 0.6, 0.8, 1], [0.2, 0, 0.2, 0.4, 0.6, 0.8], [0.4, 0.2, 0, 0.2, 0.4, 0.6], [0.6, 0.4, 0.2, 0, 0.2, 0.4],[0.8, 0.6, 0.4, 0.2, 0, 0.2],[1, 0.8, 0.6, 0.4, 0.2, 0]]
        
        for graph in test_samples:
            prediction, softmax = self.sess.run([self.bin_prediction, self.prob], feed_dict=graph)
            predictions.append(prediction)
            prob = softmax.tolist()[0]
            softmaxs.append(prob)
            pred_index.append(prob.index(max(prob)))
        
        for i in range(len(predictions)):
            #print("ground truth:%s, prediction:%s, softmax:%s"%(test_labels[i], predictions[i], softmaxs[i]))
            print("ground truth:%s, prediction:%s, softmax:%s"%(test_labels[i], pred_index[i], softmaxs[i]))
            corrected_points += cost_matrix[int(test_labels[i])][int(pred_index[i])]
            if test_labels[i] != pred_index[i]:
                baseline_points += 5/9
        f1_score=tf.contrib.metrics.f1_score(test_labels,pred_index)    
        print("Total accuracy: %f"%(sum(np.array(test_labels)==np.array(pred_index))/len(test_labels)))
        print("Weighted cost: %f"%(corrected_points/len(test_labels)))
        print("AUC: %f"%AUC(test_labels, softmaxs, self.label_kind))
        print("Brier Score: %f"%BrierScore(test_labels, softmaxs, self.label_kind))
    
    def make_batch_samples(self, graph_samples, all_labels):
        
        max_nodes_in_batch = self.batch_size         
        batch_samples, labels = [], []
        current_batch = []
        nodes_in_curr_batch = 0
        
        for sample_index, graph_sample in enumerate(graph_samples):
            
            num_nodes_in_sample = graph_sample[self.placeholders['node_label_indices']].shape[0]
            
            # Skip sample if it is too big
            if num_nodes_in_sample > max_nodes_in_batch:
                continue
                
            if num_nodes_in_sample + nodes_in_curr_batch < max_nodes_in_batch:
                current_batch.append(graph_sample)
                nodes_in_curr_batch += num_nodes_in_sample
                
            else:
                batch_samples.append(self.make_batch(current_batch))
                current_batch = [graph_sample]
                nodes_in_curr_batch = num_nodes_in_sample
                
            labels.append(all_labels[sample_index])
            
        if len(current_batch) > 0:
            batch_samples.append(self.make_batch(current_batch))

        return batch_samples, labels
        
        
    def make_batch(self, graph_samples):
        
        node_offset = 0
        node_reps = []
        semi_ids, semi_masks = [], []
        num_incoming_edges_per_type, num_outgoing_edges_per_type = [], []
        adj_lists = [[] for _ in range(self.ggnn_params['n_edge_types'])]
        targets = []
        
        for graph_sample in graph_samples:
            
            num_nodes_in_graph = graph_sample[self.placeholders['node_label_indices']].shape[0]
            
            node_reps.append(graph_sample[self.placeholders['node_label_indices']])
            
            semi_ids.append(graph_sample[self.placeholders['semi_ids']] + graph_sample[self.placeholders['semi_ids_mask']] * node_offset)
            
            semi_masks.append(graph_sample[self.placeholders['semi_ids_mask']])
            
            num_incoming_edges_per_type.append(graph_sample[self.placeholders['num_incoming_edges_per_type']])

            num_outgoing_edges_per_type.append(graph_sample[self.placeholders['num_outgoing_edges_per_type']])
            
            targets.append(graph_sample[self.placeholders['targets']])
            
            for i in range(self.ggnn_params['n_edge_types']):
                adj_lists[i].append(graph_sample[self.placeholders['adjacency_lists'][i]] + node_offset)
            
            node_offset += num_nodes_in_graph
            
        all_node_reps = np.vstack(node_reps)
        node_rep_mask = (all_node_reps != self.pad_token_id).astype(int)
        
        unique_label_subtokens, unique_label_indices, unique_label_inverse_indices = \
            np.unique(all_node_reps, return_index=True, return_inverse=True, axis=0)
            
        batch_sample = {
            self.placeholders['unique_node_labels']: unique_label_subtokens,
            self.placeholders['unique_node_labels_mask']: node_rep_mask[unique_label_indices],
            self.placeholders['node_label_indices']: unique_label_inverse_indices,
            self.placeholders['semi_ids']: np.vstack(semi_ids),
            self.placeholders['semi_ids_mask']: np.vstack(semi_masks),
            self.placeholders['num_incoming_edges_per_type']: np.vstack(num_incoming_edges_per_type),
            self.placeholders['num_outgoing_edges_per_type']: np.vstack(num_outgoing_edges_per_type),
            self.placeholders['num_samples_in_batch']: np.ones((1)) * len(targets),
            self.placeholders['targets']: np.vstack(targets)
        }
        
        for i in range(self.ggnn_params['n_edge_types']):
            if len(adj_lists[i]) > 0:
                adj_list = np.vstack(adj_lists[i])
            else:
                adj_list = np.zeros((0, 2), dtype=np.int32)

            batch_sample[self.placeholders['adjacency_lists'][i]] = adj_list

        return batch_sample
