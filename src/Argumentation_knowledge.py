import collections
import tensorflow as tf
import numpy
import re
import os
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import random
import ast
import numpy as np
import keras
from sklearn.metrics import hamming_loss
import math
from numpy import array
import tensorflow_hub as hub
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 
tknzr = TweetTokenizer()

try:
    import cPickle as pickle
except:
    import pickle

class Argumentation(object):
    def __init__(self, config):
        self.config = config
        self.UNK = "<unk>"
        self.word2id = None
        self.embedding_matrix=[]
        self.term2index = None
        self.index2term = None

    def build_vocabs(self, data_train, data_dev, data_test, dim_embedding, embedding_path=None):
    
        data_source = list(data_train)    
        if self.config["vocab_include_devtest"]:
            if data_dev != None:
                data_source += data_dev
            if data_test != None:
                data_source += data_test
        
        id, argument1s, target_argument1s, source_sentiment, target_sentiment, lst2, label_distribution= zip(*data_source) 
        
        wp_vocab = set(token for sent in argument1s for token in tknzr.tokenize(sent))
        wp_vocab_knw = set(token for s in lst2 for sent in s for token in tknzr.tokenize(sent))
        wp_vocab_argument2 = set(token for sent in target_argument1s for token in tknzr.tokenize(sent))
    
        wp_vocab = wp_vocab.union(wp_vocab_argument2)
        answers= set(['supporting','attacking','arguments'])
        wp_vocab = wp_vocab.union(answers)
        unk = numpy.random.uniform(-0.2, 0.2, dim_embedding)
        embeddings = {'UNK': unk}
        embed_file= open(embedding_path, 'rt')
        lines = embed_file.readlines()
        pre_embed={}
        self.term2index={}
        count=0
        
        word_counter = collections.Counter()
        
        for word in argument1s:
          for token in tknzr.tokenize(word):
                w = token
                if self.config["lowercase"] == True:
                    w = w.lower()
                if self.config["replace_digits"] == True:
                    w = re.sub(r'\d', '0', w)
                word_counter[w] += 1
                self.term2index[w] = count
                count = count + 1
                
        for word in target_argument1s:
            for token in tknzr.tokenize(word):
                w = token
                if self.config["lowercase"] == True:
                    w = w.lower()
                if self.config["replace_digits"] == True:
                    w = re.sub(r'\d', '0', w)
                word_counter[w] += 1
                self.term2index[w] = count
                count = count + 1
                
        for para in lst2:
           for sent in para:
              for token in tknzr.tokenize(sent):
                w = token
                if self.config["lowercase"] == True:
                    w = w.lower()
                if self.config["replace_digits"] == True:
                    w = re.sub(r'\d', '0', w)
                word_counter[w] += 1
                self.term2index[w] = count
                count = count + 1        
        
        self.word2id = collections.OrderedDict([(self.UNK, 0)])
        for word, count in word_counter.most_common():
            if self.config["min_word_freq"] <= 0 or count >= self.config["min_word_freq"]:
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)
        
        if embedding_path != None and self.config["vocab_only_embedded"] == True:
            self.embedding_vocab = set([self.UNK])
            with open(embedding_path, 'r') as f:
                for line in f:
                    line_parts = line.strip().split()
                    if len(line_parts) <= 2:
                        continue
                    w = line_parts[0]
                    if self.config["lowercase"] == True:
                        w = w.lower()
                    if self.config["replace_digits"] == True:
                        w = re.sub(r'\d', '0', w)
                    self.embedding_vocab.add(w)
            
            word2id_revised = collections.OrderedDict()
            for word in self.word2id:
                if word in embedding_vocab and word not in word2id_revised:
                    word2id_revised[word] = len(word2id_revised)
            self.word2id = word2id_revised
        
        self.index2term={}
        self.term2index = self.word2id
        self.index2term = {v:k for k,v in self.term2index.items()}
        print("n_words: " + str(len(list(wp_vocab)))) 
        
        
    def construct_network(self):
    
    
        #tf.reset_default_graph()
        self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
        self.argument1_lengths = tf.placeholder(tf.int32, [None], name="argument1_lengths")
        self.word_ids_knowledge = tf.placeholder(tf.int32, [None, None, None], name="word_ids_know")
        self.knowledge_lengths = tf.placeholder(tf.int32, [None,None], name="argument1_lengths_know")
        self.argument1_tokens = tf.placeholder(tf.string, [None, None], name="word_list_argument1")
        self.knowledge_tokens = tf.placeholder(tf.string, [None, None, None], name="word_list_knowledge")
        self.knowledge_max_lengths = tf.placeholder(tf.int32, [None,None], name="argument1_lengths_max_know")
        self.word_ids_argument2 = tf.placeholder(tf.int32, [None, None], name="word_ids_argument2")
        self.word_ids_answer_a = tf.placeholder(tf.int32, [None, None], name="word_ids_argument2")
        self.answers_a_lengths = tf.placeholder(tf.int32, [None], name="argument1_lengths_argument2")
        self.answers_a_tokens = tf.placeholder(tf.string, [None, None], name="words_list_argument2")
        self.word_ids_answer_b = tf.placeholder(tf.int32, [None, None], name="word_ids_argument2")
        self.answers_b_lengths = tf.placeholder(tf.int32, [None], name="argument1_lengths_argument2")
        self.answers_b_tokens = tf.placeholder(tf.string, [None, None], name="words_list_argument2")
        self.argument2_lengths = tf.placeholder(tf.int32, [None], name="argument1_lengths_argument2")
        self.argument2_tokens = tf.placeholder(tf.string, [None, None], name="words_list_argument2")
        self.argument1_labels = tf.placeholder(tf.float32, [None,None], name="argument1_labels")
        self.target_sentiment = tf.placeholder(tf.float32, [None,None], name="target_sentiment")
        self.source_sentiment = tf.placeholder(tf.float32, [None,None], name="source_sentiment")
        self.batch_size = tf.Variable(0)
        self.max_lengths = tf.placeholder(tf.int32, [None], name="max_lengths_padding")
        self.weights_path = tf.placeholder(tf.float32, [None, None], name="weights_path")
        self.learningrate = tf.placeholder(tf.float32, name="learningrate")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.loss = 0.0
        input_tensor = None
        input_vector_size = 0 
        self.initializer = None
        if self.config["initializer"] == "normal":
            self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
        elif self.config["initializer"] == "glorot":
            self.initializer = tf.glorot_uniform_initializer()
        elif self.config["initializer"] == "xavier":
            self.initializer = tf.glorot_normal_initializer()
            
############# BILSTM ###########################################
        if self.config["neural_network"]=="BILSTM":
############################## ARGUMENT 1 Self-attention BI-LSTM  ###############################################################
         zeros_initializer = tf.zeros_initializer()
         input_tensor = None
         self.word_embeddings = tf.get_variable("word_embeddings", 
               shape=[len(self.term2index), self.config["word_embedding_size"]], 
               initializer=(zeros_initializer if self.config["emb_initial_zero"] == True else self.initializer), 
               trainable=(True if self.config["train_embeddings"] == True else False))
               
         with tf.variable_scope("argument1"):
          use_elmo = self.config["elmo"]
          if use_elmo:
              elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
              
              input_tensor = elmo(inputs={"tokens": self.argument1_tokens,"sequence_len": self.argument1_lengths},signature="tokens",as_dict=True)["elmo"]
          
          else:
          	input_tensor = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids)
          	input_vector_size = self.config["word_embedding_size"]
          	self.word_representations = input_tensor
          
          
          word_lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config["word_recurrent_size"], 
            use_peepholes=self.config["lstm_use_peepholes"], 
            state_is_tuple=True, 
            initializer=self.initializer,
            reuse=False)
            
          word_lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config["word_recurrent_size"], 
            use_peepholes=self.config["lstm_use_peepholes"], 
            state_is_tuple=True, 
            initializer=self.initializer,
            reuse=False)
                
          (lstm_outputs_fw, lstm_outputs_bw), ((_, lstm_output_fw), (_, lstm_output_bw)) = tf.nn.bidirectional_dynamic_rnn(word_lstm_cell_fw, word_lstm_cell_bw, input_tensor, sequence_length=self.argument1_lengths, dtype=tf.float32, time_major=False)
           
          lstm_outputs = tf.concat([lstm_outputs_fw, lstm_outputs_bw], -1)
          dropout_word_lstm = self.config["dropout_word_lstm"] * tf.cast(self.is_training, tf.float32) + (1.0 - tf.cast(self.is_training, tf.float32))
          #lstm_outputs = tf.nn.dropout(lstm_outputs, dropout_word_lstm)
          self.lstm_outputs = lstm_outputs
          processed_tensor_argument1_1 = lstm_outputs
          processed_tensor_argument1_last = tf.reduce_max(lstm_outputs,1)
          if self.config["argument1_composition"] == "last":
                processed_tensor_argument1_last = tf.reduce_max(lstm_outputs,1)
                self.attention_weights = tf.zeros_like(self.word_ids, dtype=tf.float32)
                
          elif self.config["argument1_composition"] == "attention":
                attention_evidence = tf.layers.dense(lstm_outputs, self.config["attention_evidence_size"], activation=tf.sigmoid, kernel_initializer=self.initializer)
                attention_weights = tf.layers.dense(attention_evidence, 1, activation=None, kernel_initializer=self.initializer)
                attention_weights = tf.reshape(attention_weights, shape=tf.shape(self.word_ids))
                if self.config["attention_activation"] == "sharp":
                    attention_weights = tf.nn.softmax(attention_weights)
                elif self.config["attention_activation"] == "soft":
                    attention_weights = tf.sigmoid(attention_weights)
                elif self.config["attention_activation"] == "linear":
                    pass
                else:
                    raise ValueError("Unknown activation for attention: " + str(self.config["attention_activation"]))

                self.attention_weights_unnormalised = attention_weights
                attention_weights = tf.where(tf.sequence_mask(self.argument1_lengths), attention_weights, tf.zeros_like(attention_weights))
                attention_weights = attention_weights / tf.reduce_sum(attention_weights, 1, keep_dims=True)
                self.attention_weights_source = tf.where(tf.sequence_mask(self.argument1_lengths), self.attention_weights_unnormalised, tf.zeros_like(self.attention_weights_unnormalised) - 1e6)

                processed_tensor_1 = tf.reduce_sum(lstm_outputs * attention_weights[:,:,numpy.newaxis], 1)

          
          if self.config["hidden_layer_size"] > 0:
             if self.config["argument1_composition"] == "attention":
                processed_tensor_argument1 = tf.layers.dense(processed_tensor_1, self.config["hidden_layer_size"], activation=tf.nn.leaky_relu, kernel_initializer=self.initializer)
             elif self.config["argument1_composition"] == "last": 
               processed_tensor_argument1 = tf.layers.dense(processed_tensor, self.config["hidden_layer_size"], activation=tf.nn.relu, kernel_initializer=self.initializer)
          

######################### Argument2 Self-attention BI-LSTM ##################################################   
         with tf.variable_scope("target"):  
        
          if use_elmo:
                elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True) 
                input_tensor= elmo(inputs={"tokens": self.argument2_tokens,"sequence_len": self.argument2_lengths},signature="tokens",as_dict=True)["elmo"]
                
          else:
          	input_tensor = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids_argument2)
          	input_vector_size = self.config["word_embedding_size"]
          	self.word_representations = input_tensor
            
          
          argument2_lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config["word_recurrent_size"], 
            use_peepholes=self.config["lstm_use_peepholes"], 
            state_is_tuple=True, 
            initializer=self.initializer,
            reuse=False)
            
          argument2_lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config["word_recurrent_size"], 
            use_peepholes=self.config["lstm_use_peepholes"], 
            state_is_tuple=True, 
            initializer=self.initializer,
            reuse=False)
            
               
          (lstm_outputs_fw, lstm_outputs_bw), ((_, lstm_output_fw), (_, lstm_output_bw)) = tf.nn.bidirectional_dynamic_rnn(argument2_lstm_cell_fw, argument2_lstm_cell_bw, input_tensor, sequence_length=self.argument2_lengths, dtype=tf.float32, time_major=False)
          
          lstm_outputs = tf.concat([lstm_outputs_fw, lstm_outputs_bw], -1)
          dropout_word_lstm = self.config["dropout_word_lstm"] * tf.cast(self.is_training, tf.float32) + (1.0 - tf.cast(self.is_training, tf.float32))
          #lstm_outputs = tf.nn.dropout(lstm_outputs, dropout_word_lstm)
          self.lstm_outputs = lstm_outputs
          processed_tensor_argument2_1 = lstm_outputs
          processed_tensor_argument2_last = tf.reduce_sum(lstm_outputs,1)
          
          if self.config["argument1_composition"] == "last":
                processed_tensor_argument2_last = tf.reduce_max(lstm_outputs,1)
                self.attention_weights_unnormalised = tf.zeros_like(self.word_ids_argument2, dtype=tf.float32)
                
          elif self.config["argument1_composition"] == "attention":      
                attention_evidence = tf.layers.dense(lstm_outputs, self.config["attention_evidence_size"], activation=tf.sigmoid, kernel_initializer=self.initializer)
                attention_weights = tf.layers.dense(attention_evidence, 1, activation=None, kernel_initializer=self.initializer)
                attention_weights = tf.reshape(attention_weights, shape=tf.shape(self.word_ids_argument2))

                if self.config["attention_activation"] == "sharp":
                    attention_weights = tf.nn.softmax(attention_weights)
                elif self.config["attention_activation"] == "soft":
                    attention_weights = tf.sigmoid(attention_weights)
                elif self.config["attention_activation"] == "linear":
                    pass
                else:
                    raise ValueError("Unknown activation for attention: " + str(self.config["attention_activation"]))

                self.attention_weights_unnormalised = attention_weights
                attention_weights = tf.where(tf.sequence_mask(self.argument2_lengths), attention_weights, tf.zeros_like(attention_weights))
                attention_weights = attention_weights / tf.reduce_sum(attention_weights, 1, keep_dims=True)
                self.attention_weights_target = tf.where(tf.sequence_mask(self.argument2_lengths), self.attention_weights_unnormalised, tf.zeros_like(self.attention_weights_unnormalised) - 1e6)
                processed_tensor_argument2 = tf.reduce_sum(lstm_outputs * attention_weights[:,:,numpy.newaxis], 1)
                
          if self.config["hidden_layer_size"] > 0:
                 processed_tensor_argument2 = tf.layers.dense(processed_tensor_argument2, self.config["hidden_layer_size"], activation=tf.nn.leaky_relu, kernel_initializer=self.initializer)


################################################ CROSS ATTENTION ################################################################

        t_argument_1 = tf.transpose(processed_tensor_argument1_1, [0, 2, 1])
          
        #processed_tensor_1 = tf.concat([processed_tensor_argument1, processed_tensor_argument2], 1)
        processed_tensor_2 = tf.layers.dense(processed_tensor_argument2, self.config["hidden_layer_size"], activation=tf.nn.leaky_relu, kernel_initializer=self.initializer)

        if self.config["argument1_composition"] == "attention":
                processed_tensor_2 = tf.expand_dims(processed_tensor_2, -1) #batch, Dim, 1
                processed_tensor_2 = tf.transpose(processed_tensor_2, [0,2,1])
                attention_weights = tf.matmul(processed_tensor_2, t_arguments_1) #batch, length_of_argument1, number of Knowledge
                if self.config["attention_activation"] == "hard":
                    attention_weights = tf.exp(attention_weights)
                elif self.config["attention_activation"] == "soft":
                    attention_weights = tf.nn.sigmoid(attention_weights)
                elif self.config["attention_activation"] == "linear":
                    pass
                else:
                    raise ValueError("Unknown activation for attention: " + str(self.config["attention_activation"]))

                self.attention_weights_unnormalised = attention_weights # batch, 1, number of Knowledge
                sum_attention_weights = attention_weights
                attention_weights = attention_weights / tf.reduce_sum(sum_attention_weights, -1, keep_dims=True)
                #self.attention_weights = tf.squeeze(attention_weights)
                
                attention_weights = tf.transpose(attention_weights, [0, 2, 1])
                cross_argument_1 = processed_tensor_argument1_1 * attention_weights  
                cross_argument_1 = tf.reduce_sum(processed_tensor_knowledge, 1)
                cross_argument_1 = tf.layers.dense(processed_tensor_knowledge, self.config["hidden_layer_size"], activation=tf.nn.leaky_relu, kernel_initializer=self.initializer)

 
        t_argument_2 = tf.transpose(processed_tensor_argument2_1, [0, 2, 1])
          
        #processed_tensor_1 = tf.concat([processed_tensor_argument1, processed_tensor_argument2], 1)
        processed_tensor_1 = tf.layers.dense(processed_tensor_argument1, self.config["hidden_layer_size"], activation=tf.nn.leaky_relu, kernel_initializer=self.initializer)

        if self.config["argument1_composition"] == "attention":
                processed_tensor_1 = tf.expand_dims(processed_tensor_1, -1) #batch, Dim, 1
                processed_tensor_1 = tf.transpose(processed_tensor_1, [0,2,1])
                attention_weights = tf.matmul(processed_tensor_1, t_arguments_2) #batch, length_of_argument1, number of Knowledge
                if self.config["attention_activation"] == "hard":
                    attention_weights = tf.exp(attention_weights)
                elif self.config["attention_activation"] == "soft":
                    attention_weights = tf.nn.sigmoid(attention_weights)
                elif self.config["attention_activation"] == "linear":
                    pass
                else:
                    raise ValueError("Unknown activation for attention: " + str(self.config["attention_activation"]))

                self.attention_weights_unnormalised = attention_weights # batch, 1, number of Knowledge
                sum_attention_weights = attention_weights
                attention_weights = attention_weights / tf.reduce_sum(sum_attention_weights, -1, keep_dims=True)
                #self.attention_weights = tf.squeeze(attention_weights)
                
                attention_weights = tf.transpose(attention_weights, [0, 2, 1])
                cross_argument_2 = processed_tensor_argument1_1 * attention_weights  
                cross_argument_2 = tf.reduce_sum(processed_tensor_knowledge, 1)
                cross_argument_2 = tf.layers.dense(processed_tensor_knowledge, self.config["hidden_layer_size"], activation=tf.nn.leaky_relu, kernel_initializer=self.initializer) 


         input_representation = tf.concat([cross_argument_1, cross_argument_2, cross_argument_1 - cross_argument_2],1)
         input_representation = tf.layers.dense(input_representation, self.config["hidden_layer_size"], activation=tf.nn.leaky_relu, kernel_initializer=self.initializer) 
###################### KNOWLEDGE Bi-LSTM ####################################################
         
         with tf.variable_scope("knowledge"):          
          knowledge_input_tensor = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids_knowledge)
          input_vector_size = self.config["word_embedding_size"]
          
          s = tf.shape(knowledge_input_tensor)
          knowledge_input_tensor = tf.reshape(knowledge_input_tensor, shape=[s[0]*s[1], s[2], self.config["word_embedding_size"]])
          knowledge_lengths = tf.reshape(self.knowledge_max_lengths, shape=[s[0]*s[1]])
           
          know_lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(300, 
            use_peepholes=self.config["lstm_use_peepholes"], 
            state_is_tuple=True, 
            initializer=self.initializer,
            reuse=False)
            
          know_lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(300, 
            use_peepholes=self.config["lstm_use_peepholes"], 
            state_is_tuple=True, 
            initializer=self.initializer,
            reuse=False)
          
          char_lstm_outputs = tf.nn.bidirectional_dynamic_rnn(know_lstm_cell_fw, know_lstm_cell_bw, knowledge_input_tensor, sequence_length=knowledge_lengths, dtype=tf.float32, time_major=False)
          _, ((_, char_outputs_fw), (_, char_outputs_bw)) = char_lstm_outputs
          
          lstm_outputs = tf.concat([char_outputs_fw, char_outputs_bw], -1)       
          lstm_outputs = tf.reshape(lstm_outputs, shape=[s[0], s[1], 2*self.config["word_embedding_size"]])
          #know_lstm_outputs = tf.reshape(know_lstm_outputs, shape=[s[0], s[1], 2*self.config["word_embedding_size"]])       
          knowledge_output_vector_size = 2 * self.config["word_embedding_size"] 
          if self.config["whidden_layer_size"] > 0:
              lstm_outputs = tf.layers.dense(lstm_outputs, self.config["hidden_layer_size"], activation=tf.nn.relu, kernel_initializer=self.initializer)
          self.lstm_outputs = lstm_outputs     
          t_lstm_outputs = tf.transpose(lstm_outputs, [0, 2, 1])
          
          processed_tensor_1 = input_representation

          if self.config["argument1_composition"] == "attention":
                processed_tensor_1 = tf.expand_dims(processed_tensor_1, -1) #batch, Dim, 1
                processed_tensor_1 = tf.transpose(processed_tensor_1, [0,2,1])
                attention_weights = tf.matmul(processed_tensor_1, t_lstm_outputs) #batch, length_of_argument1, number of Knowledge
                if self.config["attention_activation"] == "hard":
                    attention_weights = tf.exp(attention_weights)
                elif self.config["attention_activation"] == "soft":
                    attention_weights = tf.nn.sigmoid(attention_weights)
                elif self.config["attention_activation"] == "linear":
                    pass
                else:
                    raise ValueError("Unknown activation for attention: " + str(self.config["attention_activation"]))

                self.attention_weights_unnormalised = attention_weights # batch, 1, number of Knowledge
                sum_attention_weights = attention_weights
                attention_weights = attention_weights / tf.reduce_sum(sum_attention_weights, -1, keep_dims=True)
                #self.attention_weights = tf.squeeze(attention_weights)
                
                attention_weights = tf.transpose(attention_weights, [0, 2, 1])
                processed_tensor_knowledge = lstm_outputs * attention_weights #+ tf.reduce_sum(attention_weights * lstm_outputs, axis=1)
                #processed_tensor_knowledge = tf.layers.dense(processed_tensor_knowledge, self.config["hidden_layer_size"], activation=tf.nn.tanh, kernel_initializer=self.initializer)        
                processed_tensor_knowledge = tf.reduce_sum(processed_tensor_knowledge, 1)
                knowledge_representation = tf.layers.dense(processed_tensor_knowledge, self.config["hidden_layer_size"], activation=tf.nn.leaky_relu, kernel_initializer=self.initializer)        
          
           
################################ CALCULATE SCORE ##############################################################
         
         if self.config["argument1_composition"] == "attention":
              
             
              z = tf.concat([input_representation , knowledge_representation], 1)
              z = tf.layers.dense(z, self.config["hidden_layer_size"], activation=tf.nn.sigmoid, kernel_initializer=self.initializer)

              softmax_w = tf.get_variable('softmax_w', shape=[100,2],initializer=tf.zeros_initializer, dtype=tf.float32)    
              
         
              
         softmax_b = tf.get_variable('softmax_b', shape=[2],initializer=tf.zeros_initializer, dtype=tf.float32)
         final_score = z*input_representation + (1-z)*knowledge_representation
         
         if self.config["knowledge_add"]==True:
              self.argument1_scores = tf.matmul(final_score, softmax_w) + softmax_b
         elif self.config["knowledge_add"]==False:
              self.argument1_scores = tf.matmul(dense_input, softmax_w) + softmax_b
         
        
          
          
##################CALCULATE SCORE#################################################################  
        
         lossy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.argument1_labels,logits=self.argument1_scores)
         
         #lossy = tf.losses.softmax_cross_entropy(self.argument1_labels, self.argument1_scores, weights=w)
         #self.argument1_scores = tf.nn.softmax(self.argument1_scores)
         #lossy = tf.losses.mean_squared_error(self.argument1_labels, self.argument1_scores)
         self.argument1_scores = tf.nn.softmax(self.argument1_scores)
         self.loss = tf.reduce_sum(lossy)
         regularizer = tf.nn.l2_loss(softmax_w)
         self.loss = tf.reduce_sum(self.loss+(0.01 * regularizer))
         self.train_op = self.construct_optimizer(self.config["opt_strategy"], self.loss, self.learningrate, self.config["clip"])
                 
              
    def construct_optimizer(self, opt_strategy, loss, learningrate, clip):
    
        optimizer = None
       
        if opt_strategy == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learningrate)
        elif opt_strategy == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learningrate)
        elif opt_strategy == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningrate)
        else:
            raise ValueError("Unknown optimisation strategy: " + str(opt_strategy))

        if clip > 0.0:
            grads, vs = zip(*optimizer.compute_gradients(loss))
            grads, gnorm  = tf.clip_by_global_norm(grads, clip)
            train_op = optimizer.apply_gradients(zip(grads, vs))
        else:
            train_op = optimizer.minimize(loss)
            
        return train_op


    def preload_word_embeddings(self, embedding_path):
        loaded_embeddings = set()
        embedding_matrix = self.session.run(self.word_embeddings)
        with open(embedding_path, 'r') as f:
            for line in f:
                line_parts = line.strip().split()
                if len(line_parts) <= 2:
                    continue
                w = line_parts[0]
                if self.config["lowercase"] == True:
                    w = w.lower()
                if self.config["replace_digits"] == True:
                    w = re.sub(r'\d', '0', w)
                if w in self.term2index and w not in loaded_embeddings:
                    word_id = self.term2index[w]
                    embedding = numpy.array(line_parts[1:])
                    embedding_matrix[word_id] = embedding
                    loaded_embeddings.add(w)            
        self.session.run(self.word_embeddings.assign(embedding_matrix))
        
    
    def translate2id(self, token, token2id, unk_token, lowercase=True, replace_digits=False):
    
        if lowercase == True:
            token = token.lower()
        if replace_digits == True:
            token = re.sub(r'\d', '0', token)
        token_id = None
        if token in token2id:
            token_id = token2id[token]
        elif unk_token != None:
            token_id = token2id[unk_token]
        else:
            raise ValueError("Unable to handle value, no UNK token: " + str(token))
        return token_id


    def create_input_dictionary_for_batch(self, batch, is_training, learningrate):
    
        max_length = 0
        max_lengths=[]
        argument1_max_lengths=[]
        knowledge_max_lengths=[]
        argument2_max_lengths=[]
        argument1s_pad = []
        knowledge_pad = []
        argument2_pad = []
        id, argument1s, target_argument1s, source_sentiment, target_sentiment, knowledge, label_distribution = zip(*batch)
        support =["support"]*len(id)
        attack = ["attack"]*len(id)
        word_ids_answers_a, answers_a_lengths, argument1_classes, answers_a_tokens = self.extract_input(support, label_distribution,0)
        word_ids_answers_b, answers_b_lengths, argument1_classes, answers_b_tokens = self.extract_input(attack, label_distribution,0)
        word_ids, argument1_length, argument1_classes, argument1_tokens = self.extract_input(argument1s, label_distribution,0)
        word_ids_knowledge, knowledge_lengths, argument1_classes, knowledge_tokens = self.extract_input(knowledge,label_distribution,1)
        word_ids_argument2, argument2_length, argument1_classes, argument2_tokens = self.extract_input(target_argument1s,label_distribution,0) 
        
        if self.config["argument1_composition"] == "last" or self.config["argument1_composition"] == "attention" :
            max_length_know = 0
            max_length_sent = 0
            max_length_argument2 = 0
            max_length_sent = max(argument1_length)
            for i in range(len(knowledge_lengths)):
               if max_length_know < max(knowledge_lengths[i]):
                      max_length_know = max(knowledge_lengths[i])
            max_length_argument2 = max(argument2_length)
            max_length = 56
            argument1_lengths=[]
            argument2_lengths=[]
            max_lengths=[]
            max_length_a = max(answers_a_lengths)
            max_length_b = max(answers_b_lengths)
            a_pad = self._make_padding(word_ids_answers_a, max_length_a)
            b_pad = self._make_padding(word_ids_answers_b, max_length_b)
            for i in range(len(argument1s)):
                max_lengths.append(max_length)
                
            for i in range(len(argument1s)):
                argument1_lengths.append(argument1_length[i]) 
 
            for i in range(len(argument1s)):
                argument2_lengths.append(argument2_length[i])
            
            for i in range(len(argument1s)):
                length = len(knowledge_lengths[i])
                knowledge_max_lengths.append([max_length_know]*length)
                
            

            knowledge_pad=[]
            argument1s_pad =  self._make_padding(word_ids, max_length_sent)
            for i in range(len(word_ids_knowledge)):
                knowledge_pad.append(self._make_padding(word_ids_knowledge[i], max_length_know))
            argument2_pad = self._make_padding(word_ids_argument2, max_length_argument2)
            input_dictionary = {self.word_ids: argument1s_pad, self.batch_size: len(argument1s), self.target_sentiment:target_sentiment, self.source_sentiment:source_sentiment, self.max_lengths: max_lengths, self.argument1_lengths: argument1_lengths, self.argument1_labels: argument1_classes, self.argument1_tokens: argument1_tokens, self.knowledge_max_lengths: knowledge_max_lengths, self.argument2_tokens: argument2_tokens, self.word_ids_argument2:  argument2_pad, self.argument2_lengths: argument2_lengths, self.learningrate: learningrate, self.is_training: is_training, self.word_ids_knowledge: word_ids_knowledge, self.knowledge_tokens:knowledge_tokens, self.knowledge_lengths: knowledge_lengths, self.word_ids_answer_a: a_pad, self.answers_a_tokens: answers_a_tokens, self.answers_a_lengths: answers_a_lengths, self.word_ids_answer_b: b_pad, self.answers_b_tokens: answers_b_tokens, self.answers_b_lengths: answers_b_lengths,}
            
        return input_dictionary        
        
    def map_word2embedding(self, sent, word_ids):
    #map word to embeddings
       sent2embed=[[]]
       for i in range(len(sent)):
            x = sent[i].split(' ')
            x = [k for k in x if k]
            for j in range(len(x)):
                sent2embed[i].append(word_ids[x[j]])      
       return sent2embed
    
    
    def _make_padding(self, sequences,maximum):
    #padding the training data
       padded = keras.preprocessing.sequence.pad_sequences(sequences,maxlen=maximum)
       return(padded)
        
    def extract_input(self,X,y,l):
        
        argument1_lengths=[]
        max_length_count=[]
        argument1_list=[]
        max_1=0
        if l ==0:
          for i in range(len(X)):
            
            x = tknzr.tokenize(X[i])
            x = [k for k in x if k]
            argument1_lengths.append(len(x))  
            
        elif l ==1:
           max_1=0       
           for i in range(len(X)):
               
            if max_1<len(X[i]):
                max_1=len(X[i])
           all_lengths=[0]*max_1
           
           for i in range(len(X)):
              for j in range(len(X[i])):
                  #x = X[i][j].split(' ')
                  x = tknzr.tokenize(X[i][j])
                  x = [k for k in x if k]        
                  all_lengths[j]=len(x)
              argument1_lengths.append(all_lengths)
              all_lengths=[0]*max_1
        max_argument1_length = max(argument1_lengths)
        argument1_classes = [[]]
        argument1_labels = numpy.zeros((len(X), 1), dtype=numpy.float32)
        if l==0:
          word_ids = numpy.zeros((len(X),max_argument1_length), dtype=numpy.int32)
          argument1_list = [[' '] * max_argument1_length for i in range(len(X))]
          for i in range(len(X)):
            #x = X[i].split(' ')
            x = tknzr.tokenize(X[i])
            x = [k for k in x if k]
            count =0
            for j in range(len(x)): 
                 argument1_list[i][j]=x[j]
                 word_ids[i][j] = self.translate2id(x[j], self.term2index, self.UNK, lowercase=self.config["lowercase"], replace_digits=self.config["replace_digits"])
                 count+=1
                 
            a = y[i]
            
            argument1_classes[i]=a
                
            if i<len(X)-1:
                argument1_classes.append([]) 
                
        elif l ==1:
         max_argument1_length = 0
         for i in range(len(X)):
            if max_argument1_length < max(argument1_lengths[i]):
                max_argument1_length = max(argument1_lengths[i])
         max_1=0       
         for i in range(len(X)):
            if max_1<len(X[i]):
                max_1=len(X[i])
         
         word_ids= numpy.zeros((len(X),max_1,max_argument1_length), dtype=numpy.int32)   
         argument1_list=[]  
         argument1_l = [[' '] * max_argument1_length for j in range(max_1)]
         for i in range(len(X)):
              argument1_list.append(argument1_l)   
         
         for i in range(len(X)):
           for j in range(len(X[i])):
             #x = X[i][j].split(' ')
             x = tknzr.tokenize(X[i][j])
             x = [k for k in x if k]
             for k in range(len(x)): 
                  word_ids[i][j][k]=self.translate2id(x[k], self.term2index, self.UNK, lowercase=self.config["lowercase"], replace_digits=self.config["replace_digits"]) 
                  argument1_list[i][j][k]=x[k]
             
           a = y[i]
           argument1_classes[i]=a      
           if i<len(X)-1:
                argument1_classes.append([])       
        return word_ids, argument1_lengths, argument1_classes, argument1_list


    def process_batch(self, data, batch, is_training, learningrate):
    
        feed_dict = self.create_input_dictionary_for_batch(batch, is_training, learningrate)
        cost, argument1_scores, attention_weights_s, attention_weights_t = self.session.run([self.loss, self.argument1_scores, self.attention_weights_source, self.attention_weights_target] + ([self.train_op] if is_training == True else []), feed_dict=feed_dict)[:4]
        token_scores_s = attention_weights_s
        token_scores_t = attention_weights_t
        return cost, argument1_scores, token_scores_s, token_scores_t
        


    def initialize_session(self):
        tf.set_random_seed(self.config["random_seed"])
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = self.config["tf_allow_growth"]
        session_config.gpu_options.per_process_gpu_memory_fraction = self.config["tf_per_process_gpu_memory_fraction"]
        self.session = tf.Session(config=session_config)
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)


    def get_parameter_count(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters


    def get_parameter_count_without_word_embeddings(self):
        shape = self.word_embeddings.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        return self.get_parameter_count() - variable_parameters


    def save(self, filename):
        dump = {}
        dump["config"] = self.config
        dump["UNK"] = self.UNK
        dump["word2id"] = self.word2id
        dump["params"] = {}
        for variable in tf.global_variables():
            assert(variable.name not in dump["params"]), "Error: variable with this name already exists" + str(variable.name)
            dump["params"][variable.name] = self.session.run(variable)
        with open(filename, 'wb') as f:
            pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def load(filename, new_config=None):
        with open(filename, 'rb') as f:
            dump = pickle.load(f)

            # for safety, so we don't overwrite old models
            dump["config"]["save"] = None

            # we use the saved config, except for values that are present in the new config
            if new_config != None:
                for key in new_config:
                    dump["config"][key] = new_config[key]

            labeler = Argumentation(dump["config"])
            labeler.UNK = dump["UNK"]
            #labeler.term2index = dump["term2index"]
            labeler.term2index = dump["word2id"]

            labeler.construct_network()
            labeler.initialize_session()

            labeler.load_params(filename)

            return labeler


    def load_params(self, filename):
        with open(filename, 'rb') as f:
            dump = pickle.load(f)

            for variable in tf.global_variables():
                assert(variable.name in dump["params"]), "Variable not in dump: " + str(variable.name)
                assert(variable.shape == dump["params"][variable.name].shape), "Variable shape not as expected: " + str(variable.name) + " " + str(variable.shape) + " " + str(dump["params"][variable.name].shape)
                value = numpy.asarray(dump["params"][variable.name])
                self.session.run(variable.assign(value))
