import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, RepeatVector, Bidirectional
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, CuDNNGRU


class Encoder(Layer):
    def __init__(self, hiddens=[32], cell_type='LSTM', bidirectional=False, distributed=True, **kwargs):
        super(Encoder, self).__init__(name="Encoder", **kwargs)
        self.hiddens = hiddens
        cell = {'LSTM': CuDNNLSTM, 'GRU': CuDNNGRU}[cell_type]
        self.layers = []
        for i in range(len(hiddens)):
            if bidirectional:
                rnn = Bidirectional(cell(units=hiddens[0], return_sequences=(i!=len(hiddens)-1) or distributed, return_state=(i==len(hiddens)-1)))
            else:
                rnn = cell(units=hiddens[0], return_sequences=(i!=len(hiddens)-1) or distributed, return_state=(i==len(hiddens)-1))
            self.layers.append(rnn)

    def call(self, inputs):
        h = inputs
        for rnn in self.layers:
            h = rnn(h)
        return h


class Decoder(Layer):
    def __init__(self, hiddens=[32], cell_type='LSTM', **kwargs):
        super(Decoder, self).__init__(name="Decoder", **kwargs)
        self.hiddens = hiddens
        cell = {'LSTM': CuDNNLSTM, 'GRU': CuDNNGRU}[cell_type]
        self.layers = []
        for i, units in enumerate(hiddens):
            rnn = cell(units=units, return_state=True, return_sequences=(i!=len(hiddens)-1))
            self.layers.append(rnn)

    def call(self, inputs, states):
        outputs, hiddens_s, cells_s = [], [], []
        h = inputs
        for i, rnn in enumerate(self.layers):
            output, hidden_s, cell_s = rnn(h, initial_state=states[i])
            h = output
            hiddens_s.append(hidden_s)
            cells_s.append(cell_s)
            outputs.append(output)
        return outputs, hiddens_s, cells_s
    
    
class LSTM_ED(Layer):
    def __init__(self, 
                 n_features, 
                 encoder_hiddens=[100, 100], 
                 decoder_hiddens=[100, 100], 
                 steps=12,
                 bidirectional=False,
                 cell_type='LSTM', **kwargs):
        super(LSTM_ED, self).__init__(name="LSTM_ED", **kwargs)
        self.steps = steps
        cell = {'LSTM': CuDNNLSTM, 'GRU': CuDNNGRU}[cell_type]
        self.encoder =[]
        for i, units in enumerate(encoder_hiddens):
            if bidirectional:
                self.encoder.append(Bidirectional(cell(units=units, return_sequences=(i!=len(encoder_hiddens)-1))))
            else:
                self.encoder.append(cell(units=units, return_sequences=(i!=len(encoder_hiddens)-1)))

        self.decoder = []
        for units in decoder_hiddens:
            self.decoder.append(cell(units=units, return_sequences=True))
        # self.decoder_dense = TimeDistributed(Dense(units=n_features))
        self.decoder_dense = Dense(units=n_features)
    
    def call(self, X_inputs, X_target=None, target_covars=None, training=True):
        h = X_inputs
        for layer in self.encoder:
            h = layer(h)
        ctx_vec = h
        ctx_vec = RepeatVector(self.steps)(ctx_vec)
        
        if target_covars is not None:
            decoder_inputs = tf.concat([ctx_vec, target_covars], axis=-1)
        else:
            decoder_inputs = ctx_vec
        
        h = decoder_inputs
        for layer in self.decoder:
            h = layer(h)
        decoder_outputs = h
        
        decoder_outputs = self.decoder_dense(decoder_outputs)
        return decoder_outputs