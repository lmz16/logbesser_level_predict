import tensorflow as tf
from graph_pb2 import Graph
from dpu_utils.tfmodels import SparseGGNN
from data_processing.sample_inf_processing import SampleMetaInformation, CorpusMetaInformation
import numpy as np
import os
from data_processing import rnn_log_processing
from data_processing.graph_features import get_used_edges_type
from random import shuffle
from utils.utils import compute_f1_score
import json
import datetime

class Model:

    def __init__(self, mode, task_id, vocabulary):

        # Initialize parameter values
        self.max_node_seq_len = 32                          # Maximum number of node subtokens
        self.max_var_seq_len = 16                           # Maximum number of variable subtokens
        self.token_len = 30
        self.learning_rate = 0.001
        self.vocabulary = vocabulary
        self.voc_size = len(vocabulary)
        self.pad_token_id = self.vocabulary.get_id_or_unk(self.vocabulary.get_pad())
        self.embedding_size1 = 64
        self.embedding_size2 = 32
        self.embedding_size3 = 16
        self.embedding_size4 = 6
        self.hidden_layer_num = 3
        self.dropout = 1.0
        self.label_kind = 6
        self.batch_size = 1
        self.enable_batching = False
        
        if mode == "infer":
            self.enable_batching = False

        if mode != 'train' and mode != 'infer':
            raise ValueError("Invalid mode. Please specify \'train\' or \'infer\'...")
            
        if self.enable_batching:
            self.batch_size = 20

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


    def make_model(self):

        # Create inputs and compute initial node representations
        self.make_inputs()
        self.get_initial_node_representation()

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.embedding_size1, forget_bias=0.0, state_is_tuple=True)
            
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(self.hidden_layer_num)], state_is_tuple=True)
        
        self._initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        
        out_put = 0
        state = self._initial_state
        with tf.variable_scope("LSTM_layer"):
            for time_step in range(self.token_len):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(tf.reshape(self.token_representations[time_step, :], [1, -1]), state)
                out_put = cell_output
                 
        self.output = tf.reshape(out_put, [-1, 1])

        mid_result1 = tf.layers.dense(inputs=self.output, units=self.embedding_size3, use_bias=True, bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu)
#        mid_result2 = tf.layers.dense(inputs=mid_result1, units=self.embedding_size2, activation=tf.nn.relu)
#        mid_result3 = tf.layers.dense(inputs=mid_result2, units=self.embedding_size3, activation=tf.nn.relu)
##        if self.model_type == "c":
##            self.bin_prediction = tf.layers.dense(inputs=mid_result3, units=6)
##            self.prob = tf.nn.softmax(self.bin_prediction)
##        elif self.model_type == "r":
##            mid_result4 = tf.layers.dense(inputs=mid_result3, units=self.embedding_size4, activation=tf.nn.relu)
##            self.float_prediction = tf.layers.dense(inputs=mid_result4, units=1, activation=tf.nn.sigmoid)
##            self.prob = 10.0 * self.float_prediction - 2.5
        self.bin_prediction = tf.layers.dense(inputs=mid_result1, units=6)
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
        self.placeholders['token_sequence'] = tf.placeholder(name='token_sequence', shape=[None, self.token_len, self.max_node_seq_len], dtype=tf.int32 )
        self.placeholders['token_sequence_mask'] = tf.placeholder(name='token_sequence_mask', shape=[None, self.token_len, self.max_node_seq_len], dtype=tf.float32)
        self.placeholders['targets'] = tf.placeholder(dtype=tf.float32, shape=[None, self.label_kind], name='target')
        
        self.placeholders['num_samples_in_batch'] = tf.placeholder(dtype=tf.float32, shape=(1), name='num_samples_in_batch')


    def get_initial_node_representation(self):

        # Compute the embedding of input node sub-tokens
        self.embedding_encoder = tf.get_variable('embedding_encoder', [self.voc_size, self.embedding_size1])
        subtoken_embedding = tf.nn.embedding_lookup(params=self.embedding_encoder, ids=tf.reshape(self.placeholders['token_sequence'], [-1, self.max_node_seq_len]))
        subtoken_ids_mask = tf.reshape(self.placeholders['token_sequence_mask'], [-1, self.max_node_seq_len, 1])
        subtoken_embedding = subtoken_ids_mask * subtoken_embedding
        token_representations = tf.reduce_sum(subtoken_embedding, axis=1)
        num_subtokens = tf.reduce_sum(subtoken_ids_mask, axis=1)
        self.token_representations = token_representations / num_subtokens
        

    
    def train(self, train_path, val_path, n_epochs, checkpoint_path):

        train_samples, train_labels = self.get_samples(train_path)
        print("Extracted training samples... ", len(train_samples))

        with self.graph.as_default():

            for epoch in range(n_epochs):

                loss = 0
                count = 0

                for sample in train_samples:
                    loss += self.sess.run([self.train_loss, self.train_step], feed_dict=sample)[0]/sample[self.placeholders['num_samples_in_batch']]
#                    #print("label: %s, prediction: %s" %(train_labels[count], self.sess.run(self.bin_prediction, feed_dict=graph)))
                    count += 1

                print("Average Epoch Loss:", (loss/len(train_samples)))
                print("Epoch: ", epoch + 1, "/", n_epochs)
                print("---------------------------------------------")
#
#
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
        shuffle(zipped)
        graph_samples, labels = zip(*zipped)
        
        if self.enable_batching:
            graph_samples, labels = self.make_batch_samples(graph_samples, labels)

        return graph_samples, labels

    
    def create_samples(self, filepath):

        with open(filepath, "rb") as f:

            g = Graph()
            g.ParseFromString(f.read())
            
            true_labels = []

            rnn_len = self.token_len
            
            seq_samples = rnn_log_processing.get_log_samples(g, self.max_node_seq_len, self.pad_token_id, self.vocabulary, rnn_len)

            samples, labels, slot_labels = [], [], []
            
            try:
                with open(os.path.splitext(filepath)[0].replace("java", "json"), "r") as f_:
                    labels, slot_labels = json.load(f_)
                    if len(slot_labels) != len(seq_samples) or len(labels) != sum(slot_labels):
                        print("Error: labels and samples don't match, num of labels: %d, num of samples: %d, filename: %s" % (len(labels), len(seq_samples), filepath))
                        os.system("rm -f "+filepath[:-11]+"*")
            except FileNotFoundError as e:
                print("Warning: file not found. It's ok if you are not training the model")
                labels = [self.label_kind - 1] * len(seq_samples)
                slot_labels = [1] * len(seq_samples)

            count = 0

            for i in range(len(seq_samples)):
                if slot_labels[i] != 0:
                    new_sample = self.create_sample(seq_samples[i], labels[count])
                    samples.append(new_sample)
                    true_labels.append(labels[count])
                    count += 1

            return samples, true_labels

    def create_sample(self, node_representation, label):
        
        target = np.zeros((1, self.label_kind))
        target[0, label] = 1
        
        node_rep_mask = (node_representation != self.pad_token_id).astype(int)

        seq_sample = {
            self.placeholders['token_sequence']: node_representation,
            self.placeholders['token_sequence_mask']: node_rep_mask,
            self.placeholders['targets']: target,
            self.placeholders['num_samples_in_batch']: np.ones((1))
        }

        return seq_sample
        
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
    
    def make_batch_samples(self, graph_samples, all_labels):
        
        max_nodes_in_batch = self.batch_size         
        batch_samples, labels = [], []
        current_batch = []
        
        for sample_index, graph_sample in enumerate(graph_samples):
            
            current_batch.append(graph_sample)
            if len(current_batch) == self.batch_size:
                batch_samples.append(self.make_batch(current_batch))
                current_batch = []
            labels.append(all_labels[sample_index])
            
        if len(current_batch) > 0:
            batch_samples.append(self.make_batch(current_batch))

        return batch_samples, labels
        
        
    def make_batch(self, graph_samples):
        
        node_offset = 0
        node_reps = []
        targets = []
        
        for graph_sample in graph_samples:
            
            num_nodes_in_graph = graph_sample[self.placeholders['node_label_indices']].shape[0]
            
            node_reps.append(graph_sample[self.placeholders['node_label_indices']])
            
            semi_ids.append(graph_sample[self.placeholders['semi_ids']] + graph_sample[self.placeholders['semi_ids_mask']] * node_offset)
            
            semi_masks.append(graph_sample[self.placeholders['semi_ids_mask']])
            
            targets.append(graph_sample[self.placeholders['targets']])
            
            node_offset += num_nodes_in_graph
            
        all_node_reps = np.vstack(node_reps)
        node_rep_mask = (all_node_reps != self.pad_token_id).astype(int)
        
        unique_label_subtokens, unique_label_indices, unique_label_inverse_indices = \
            np.unique(all_node_reps, return_index=True, return_inverse=True, axis=0)
            
        batch_sample = {
            self.placeholders['unique_node_labels']: unique_label_subtokens,
            self.placeholders['unique_node_labels_mask']: node_rep_mask[unique_label_indices],
            self.placeholders['num_samples_in_batch']: np.ones((1)) * len(targets),
            self.placeholders['targets']: np.vstack(targets)
        }

        return batch_sample
