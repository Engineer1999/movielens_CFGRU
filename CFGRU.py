import numpy as np
import time
import math
import pandas as pd
import random
import os
import progressbar
import tensorflow as tf
import matplotlib.pyplot as plt
from Data_prep import standard_padding, get_x_y_sequences
from Helpers import TimingCallback
K = tf.keras.backend

class CFGRU:

    def __init__(self, total_users, total_items, params):
        self.total_users = total_users
        self.total_items = total_items
        self.params = params
        self.epochs = params['epochs']
        self.batch_size = params['BATCH_SIZE']
        self.learning_rate = params['learning_rate']
        self.delta = params['delta']
        self.max_seq_len = params['max_seq_len']
        self.embedding_dim = params['embedding_dim']
        self.rnn_units = params['rnn_units']
        self.ckpt_dir = params['ckpt_dir']
        self.pad_value = params['pad_value']
        self.history = {}
        self.diversity_bias = []
     

    def build_model(self, ckpt_dir='', return_sequences=True, initializer='glorot_uniform', summary=True):
        """
        Building a sequential LSTM model in Keras
        :param return sequences: whether return sequences has to be True in the sequential RNN model
        :param ckpt_dir: Location for storing the checkpoints
        :param initializer: Which weight initializer to use
        :param summary: True => print model.summary()
        :param return_sequences: True when training, False when predicitng next item
        :return: model of type tf.keras.model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.total_items + 1, #+1 if masking value is total_items
                                      self.embedding_dim,
                                      batch_input_shape=[self.batch_size, None]),

            tf.keras.layers.Masking(mask_value=self.total_items),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.GRU(units=64,
                                 return_sequences=return_sequences,
                                 stateful=False,  # Reset cell states with each batch
                                 recurrent_initializer=initializer),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.total_items, activation='softmax')
        ])
        
        if len(ckpt_dir) > 0:
            print('model loaded')
            model.load_weights(ckpt_dir).expect_partial()
            #print(model.summary())
        
        self.model = model
        
        if summary:
            print(model.summary())


    def create_diversity_bias(self, train_set):
        """
        Pre-calculates the diversity bias needed in the diversity_bias_loss, stores db in self.diversity_bias
        :param train_set: the train_set as a pandas df: user_id, item_id, datetime
        """
        item_id_bins = np.zeros((1, self.total_items+1), np.float32)
        item_counts = train_set.groupby('item_id')['user_id'].count().sort_values(ascending=False)
        bins = np.logspace(np.log10(item_counts.max()), np.log10(1), 11)
        item_counts.index, np.digitize([item_counts],bins)

        for item_id, count  in zip(item_counts.index, list(item_counts)):
            item_bin = np.digitize([count],bins)
            item_id_bins[0,int(item_id)] = item_bin

        diversity_bias = tf.Variable(np.exp(item_id_bins[0] * -self.delta))
        
        self.diversity_bias = diversity_bias


    def diversity_bias_loss(self):
        """
        (decorator) Calculates Categorical Cross Entropy Loss divided by the diversity bias (self.diversity_bias created in self.create_diversity_bias)as defined in Paper 1
        :return: categorical cross entropy loss function adjusted by the diversity bias
        """
        def loss(labels, logits):
            labels = tf.dtypes.cast(labels, tf.int32)
            oh_labels = K.one_hot(labels, self.total_items)
            standard_loss = tf.keras.losses.categorical_crossentropy(oh_labels, logits, from_logits=True)
            label_weights = tf.gather(self.diversity_bias, labels, axis=0)
            db_loss = tf.math.multiply(standard_loss, label_weights)
            return db_loss
        return loss


        
        
    def create_seq_batch_tf_dataset(self, df, shift=1, stats=True, drop_remainder=True):
        """
        :param df: pandas df where each row consists of user_id, item_id and datetime (chronologically)
        :param shift: how much to shift the x sequences
        :param max_seq_len: maximum sequence length
        :param drop_remainder: drop remainder as in tf.dataset.batch
        :return:
        """
        user_sequences_x, user_sequences_y, median = get_x_y_sequences(df, shift, stats=stats)
        sequences_data_x = standard_padding(user_sequences_x, self.max_seq_len, pad_value=self.pad_value, stats=stats)
        sequences_data_y = standard_padding(user_sequences_y, self.max_seq_len, pad_value=self.pad_value, stats=stats)

        dataset = tf.data.Dataset.zip((sequences_data_x, sequences_data_y))
        dataset = dataset.batch(self.batch_size, drop_remainder=drop_remainder)

        return dataset

    
    def data_split(self, df, val=False):
        """
        Split df according to self.val_users, self.test_user
        :param df: pandas df where each row consists of user_id, item_id and datetime (chronologically)
        """
        if val:
            n = self.val_users
        else: 
            n = self.test_users
            
        users_ids = np.random.choice(df['user_id'].unique(), n, replace=False)
        n_set = df[df['user_id'].isin(users_ids)]
        remaining_set = df.drop(n_set.index)
        return remaining_set, n_set

    
    def get_predictions(self, train_set, test_set, left_out_items, batch_size, rank_at, ckpt_dir='', summary=False, exclude_already_seen=False):
        """
        Uses the stored Keras LSTM model with batch size set to None to predict the rest of the sequences from the data per user.
        Finally creates predictions_df where each row represents user, a list pred_items_ranked and a list containing true_ids
        from the left_out df
        :param train_set: pandas df where each row consists of user_id, item_id and datetime (chronologically) used in training
        :param test_set: pandas df where each row consists of user_id, item_id and datetime (chronologically) to be used now (contains all but 1 item of the test users)
        :param left_out_items: pandas df where each row consists of user_id, item_id and datetime (chronologically) with the held-out items
        :param rank_at: maximum rank to compute the metrics on
        :param batch_size: batch_size==number of test users
        :param summary: True => print model.summary()
        :param exclude_already_seen: whether to exclude the items already seen in the train_set when predicting
        :return: pandas df where each row represents a user, the columns represent: pred_items_ranked at rank_at,
                 true_id extracted from test_set (as input for Evaluation.get_metrics
        """
        self.batch_size = None
        #self.model = model_final
        self.build_model(ckpt_dir=ckpt_dir, return_sequences=False, summary=summary)
        #print(model.summary())
        n_batches = int(len(left_out_items) / batch_size)
        data_sequences, _, _ = get_x_y_sequences(test_set, stats=False)
        data_seqs_padded = standard_padding(data_sequences, self.max_seq_len, self.pad_value, eval=True, stats=False)
        data_seqs_splits = np.array_split(data_seqs_padded, n_batches, axis=0)
        if exclude_already_seen:
            already_seen = pd.concat([train_set, test_set]).groupby('user_id')['item_id'].apply(list)
            
        # Get True items
        test_left_out_items = left_out_items.groupby('user_id')['item_id'].apply(list)
        
        # Extend final predictions with predictions made on batches
        preds = []
        for split in data_seqs_splits:
            preds.extend(self.model.predict(split))
        
        # Exclude alredy seen items
        final_preds = []
        
        for user, pred in zip(test_left_out_items.index, preds):
            if exclude_already_seen:
                pred[already_seen[user]] = -np.inf
            ids = np.argpartition(pred, -rank_at)[-rank_at:]
            final_preds.append(ids[np.argsort(pred[ids][::-1])])
            
        preds_df = pd.DataFrame(list(zip(test_left_out_items.index, final_preds, list(test_left_out_items))),
                                columns=['user', 'pred_items_ranked', 'true_id'])

        return preds_df
