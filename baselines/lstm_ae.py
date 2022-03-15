import os, sys
sys.path.insert(1, '../')
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
from load_data import batch_data, get_validation_data
from utils import dict_to_str
from sklearn.metrics import r2_score, mean_squared_error
from utils import *


class Encoder(layers.Layer):
    def __init__(self, time_step, x_dim, encoder_hiddens, z_dims, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.encoder_inputs = keras.Input(shape=(time_step, x_dim))
        self.encoder_lstms = []
        for i, units in enumerate(encoder_hiddens):
            self.encoder_lstms.append(layers.LSTM(units, return_sequences=True, activation='tanh', name='encoder_lstm_%d' % i))
        self.z = layers.LSTM(z_dims, return_sequences=True, activation='tanh', name='z')

    def call(self, inputs):
        hidden = inputs
        for lyer in self.encoder_lstms:
            hidden = lyer(hidden)
        z = self.z(hidden)
        return z
    
    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'name': self.name,
            'z': self.z.get_config()
        })
        return config


class Decoder(layers.Layer):
    def __init__(self, time_step, x_dim, decoder_hiddens, z_dims, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        
        self.decoder_lstms = [layers.LSTM(z_dims, return_sequences=True, activation='tanh', name='z')]
        for i, units in enumerate(decoder_hiddens):
            self.decoder_lstms.append(layers.LSTM(units, return_sequences=True, activation='tanh', name='decoder_lstm_%d' % i))    
        self.rec_x = layers.TimeDistributed(layers.Dense(x_dim, name='rec_x'))
    
    def call(self, inputs):
        hidden = inputs
        for lyer in self.decoder_lstms:
            hidden = lyer(hidden) 
        rec_x = self.rec_x(hidden)
        return rec_x
    
    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'name': self.name
        })
        return config
    

class AE(Model):
    def __init__(self, 
                 n_features, 
                 encoder_hiddens=[64, 32], 
                 decoder_hiddens=[32, 64], 
                 latent_dims=10,
                 lr=1e-3, 
                 sw_len=100, 
                 steps=10, 
                 model_dir=''):
        super(AE, self).__init__()
        self.steps = steps
        self.loss_tracker = keras.metrics.Mean(name="loss")
        learning_rate_fn = tfk.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr, 
                                                                    decay_steps=50, 
                                                                    end_learning_rate=1e-6)
        self.optimizer = tf.optimizers.Adam(learning_rate_fn)
        self.encoder = Encoder(sw_len, n_features, encoder_hiddens, latent_dims, name='encoder')
        self.decoder = Decoder(sw_len, n_features, decoder_hiddens, latent_dims, name='decoder')
        
    @property
    def metrics(self):
        return [self.loss_tracker]
    
    def call(self, inputs):
        z = self.encoder(inputs)
        rec_x = self.decoder(z)
        return rec_x

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            rec_x = self(x, training=True)
            loss = tf.losses.mean_squared_error(x, rec_x)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @tf.function
    def test_step(self, x):
        rec_x = self(x, training=False)
        loss = tf.losses.mean_squared_error(x, rec_x)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    def get_score(self, x):
        rec_x = self(x, training=False)
        loss = tf.losses.mean_squared_error(x, rec_x).numpy()
        loss = flatten_loss_1d(loss, self.steps, metric='mean')
        return loss