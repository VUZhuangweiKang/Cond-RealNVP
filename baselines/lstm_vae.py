"""
cite: https://github.com/paya54/Anomaly_Detect_LSTM_VAE
"""

import os, sys
sys.path.insert(1, '../')
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import tensorflow.keras as tfk
from tensorflow.keras import Input, layers, Model
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import mean_squared_error as mse
from load_data import batch_data, get_validation_data
from utils import dict_to_str
from sklearn.metrics import r2_score, mean_squared_error
from utils import *


class Sampling(layers.Layer):
    def __init__(self, name='sampling_z'):
        super(Sampling, self).__init__(name=name)
    
    def call(self, inputs):
        mu, logvar = inputs
        batch_size, latent_dim = tf.shape(mu)[0], tf.shape(mu)[1]
        sigma = K.exp(logvar * 0.5)
        epsilon = K.random_normal(shape=(batch_size, latent_dim))
        return mu + epsilon * sigma
    
    def get_config(self):
        config = super(Sampling, self).get_config()
        config.update({'name': self.name})
        return config

class Encoder(layers.Layer):
    def __init__(self, time_step, x_dim, encoder_hiddens, z_dim, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.encoder_inputs = keras.Input(shape=(time_step, x_dim))
        self.encoder_lstms = []
        for i, units in enumerate(encoder_hiddens):
            self.encoder_lstms.append(layers.LSTM(units, return_sequences=(i!=len(encoder_hiddens)-1), activation='tanh', name='encoder_lstm_%d' % i))
        self.z_mean = layers.Dense(z_dim, name='z_mean')
        self.z_logvar = layers.Dense(z_dim, name='z_log_var')
        self.z_sample = Sampling()
    
    def call(self, inputs):
        hidden = inputs
        for lyer in self.encoder_lstms:
            hidden = lyer(hidden)
        mu_z = self.z_mean(hidden)
        logvar_z = self.z_logvar(hidden)
        z = self.z_sample((mu_z, logvar_z))
        return mu_z, logvar_z, z
    
    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'name': self.name,
            'z_sample': self.z_sample.get_config()
        })
        return config


class Decoder(layers.Layer):
    def __init__(self, time_step, x_dim, decoder_hiddens, z_dim, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.z_inputs = layers.RepeatVector(time_step)
        self.decoder_lstms = []
        for i, units in enumerate(decoder_hiddens):
            self.decoder_lstms.append(layers.LSTM(units, return_sequences=True, activation='tanh', name='decoder_lstm_%d' % i))    
        self.x_mean = layers.TimeDistributed(layers.Dense(x_dim, name='x_mean'))
        self.x_sigma = layers.Dense(x_dim, name='x_sigma', activation='tanh')
    
    def call(self, inputs):
        hidden = self.z_inputs(inputs)
        for lyer in self.decoder_lstms:
            hidden = lyer(hidden) 
        mu_x = self.x_mean(hidden)
        sigma_x = self.x_sigma(hidden)
        return mu_x, sigma_x
    
    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'name': self.name
        })
        return config
    

class VAE(Model):
    def __init__(self, n_features, 
                 encoder_hiddens=[64], 
                 decoder_hiddens=[64], 
                 latent_dim=2, 
                 lr=1e-3, 
                 verbose=1, 
                 beta=1.0, 
                 capacity=0.0, 
                 sw_len=100, 
                 steps=10, 
                 batch_size=64,
                 model_dir=''):
        super(VAE, self).__init__()
        self.n_features = n_features
        self.sw_len = sw_len
        self.steps = steps
        
        self.batch_size = batch_size

        self.latent_dim = latent_dim
        self.beta = beta
        self.capacity = capacity

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
        self.loss = self.vae_loss
        learning_rate_fn = tfk.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr, 
                                                                    decay_steps=50, 
                                                                    end_learning_rate=1e-8)
        self.optimizer = tf.optimizers.Adam(learning_rate_fn)
        
        self.encoder = Encoder(self.sw_len, self.n_features, encoder_hiddens, self.latent_dim, name='encoder')
        self.decoder = Decoder(self.sw_len, self.n_features, decoder_hiddens, self.latent_dim, name='decoder')
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def call(self, inputs):
        mu_z, logvar_z, z = self.encoder(inputs)
        mu_x, sigma_x = self.decoder(z)
        
        dist = tfp.distributions.MultivariateNormalDiag(loc=mu_x, scale_diag=tf.abs(sigma_x))
        log_px = -dist.log_prob(inputs)

        return mu_z, logvar_z, mu_x, sigma_x, log_px

    def mean_log_likelihood(self, log_px):
        log_px = K.reshape(log_px, shape=(log_px.shape[0], log_px.shape[2]))
        mean_log_px = K.mean(log_px, axis=1)
        return K.mean(mean_log_px, axis=0)
    
    def vae_loss(self, x, mu_x, mu_z, logvar_z):
        reconst_loss = mse(x, mu_x) * self.sw_len
        kl_loss = -0.5 * K.mean(1 + logvar_z - K.square(mu_z) - K.exp(logvar_z))
        total_loss = reconst_loss + self.beta*(kl_loss-self.capacity)
        return reconst_loss, kl_loss, total_loss
        
    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            mu_z, logvar_z, mu_x, sigma_x, log_px = self(x, training=True)
            reconst_loss, kl_loss, total_loss = self.vae_loss(x, mu_x, mu_z, logvar_z)      
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconst_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    @tf.function
    def test_step(self, x):
        mu_z, logvar_z, mu_x, sigma_x, log_px = self(x, training=False)
        reconst_loss, kl_loss, total_loss = self.vae_loss(x, mu_x, mu_z, logvar_z)      
        self.total_loss_tracker.update_state(total_loss)
        return {            
            "loss": self.total_loss_tracker.result()
        }