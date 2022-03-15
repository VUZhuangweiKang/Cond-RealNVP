import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as tfk
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.models import Model
from load_data import *
from lstm_models import *


tfd = tfp.distributions
np.random.seed(123456)
tf.random.set_seed(123456)


class BatchNormLayer(Layer):
    """
    Compute `Y = g(X) s.t. X = g^-1(Y) = (Y - mean(Y)) / std(Y)`

    Inverse pass:
        1. Compute the current batch mean and standard deviation of the data distribution x.
        2. Update running mean and standard deviation
        3. Batch normalize the data using current mean/std

    Forward pass:
        1. Use running mean and standard deviation to un-normalize the data distribution.
    """

    def __init__(self, lid, n_features, eps=1e-5, decay=.9):
        super(BatchNormLayer, self).__init__(name='BN%d'%lid)
        self.eps = eps
        self.decay = decay

        # beta, gamma are learnable parameters
        self.beta = tf.Variable(initial_value=tf.zeros(shape=(1, n_features), dtype=tf.float32), name='beta', trainable=True)
        self.gamma = tf.Variable(initial_value=tf.ones(shape=(1, n_features), dtype=tf.float32), name='gamma', trainable=True)

        self.moving_mean = tf.Variable(initial_value=tf.zeros(shape=(1, n_features), dtype=tf.float32), name='moving_mean', trainable=False)
        self.moving_var = tf.Variable(initial_value=tf.ones(shape=(1, n_features), dtype=tf.float32), name='moving_var', trainable=False)
        
    def forward(self, x, condition_input):
        y = (x - self.beta) * tf.exp(-self.gamma) * tf.sqrt(self.moving_var+self.eps) + self.moving_mean
        return y
    
    def call(self, inputs, *args, **kwargs):
        """ x = (y - mu) * sqrt(sigma**2) * gamma + beta """
        y, condition_input = inputs   
        
        if kwargs['training']:
            # statistics of current minibatch
            mean, var = tf.nn.moments(y, axes=[0], keepdims=True)
        
            # update train statistics via exponential moving average
            self.moving_mean.assign_sub(self.decay * (self.moving_mean - mean))
            self.moving_var.assign_sub(self.decay * (self.moving_var - var))    
        else:
            mean, var = self.moving_mean, self.moving_var

        x = (y - mean) * 1. / tf.sqrt(var + self.eps) * tf.exp(self.gamma) + self.beta
        ldj = 0.5 * tf.math.log(var + self.eps) - self.gamma
        return x, -ldj


class CouplingLayer(Layer):
    def __init__(self, lid, n_features, mask, st_units=[100]):
        super(CouplingLayer, self).__init__(name='CouplingLayer%d'%lid)
        self.mask = tf.cast(mask, tf.float32)
        self.snet = []
        self.tnet = []
        
        for units in st_units:
            self.snet.append(Dense(units, activation='tanh'))
        self.snet.append((Dense(n_features, activation='tanh')))
        
        for units in st_units:
            self.tnet.append(Dense(units, activation='relu'))
        self.tnet.append(Dense(n_features))

    def conditioner(self, x, conditions):
        if conditions is not None:
            x = tf.concat((x, conditions), axis=-1)
        
        s = x
        for model in self.snet:
            s = model(s)
        log_scale = s

        t = x
        for model in self.tnet:
            t = model(t)
        shift = t
        return log_scale, shift

    # Forward propagation: generative direction --> sampling
    def forward(self, x, condition_input):
        x_masked = x * self.mask
        reversed_mask = 1 - self.mask
        s, t = self.conditioner(x_masked, condition_input)
        s *= reversed_mask
        t *= reversed_mask        
        y = x_masked + reversed_mask * (x * tf.exp(s) + t)
        return y

    # Inverse propagation: normalizing direction --> density estimation
    def call(self, inputs, *args, **kwargs):
        y, condition_input = inputs
        y_masked = y * self.mask
        reversed_mask = 1 - self.mask
        s, t = self.conditioner(y_masked, condition_input)
        s *= reversed_mask
        t *= reversed_mask
        x = reversed_mask * (y - t) * tf.exp(-s) + y_masked
        return x, -s


class CondRealNVP(Model):
    def __init__(self, 
                 steps, 
                 n_features, 
                 cond_size, 
                 num_blocks=3, 
                 encoder_units=[100, 100],
                 decoder_units=[100, 100],
                 rnn=None,
                 rnn_cell_type='LSTM',
                 bidirectional=False,
                 attention=False,
                 teacher_fc_ratio=0.,
                 st_units=[100], 
                 batch_norm=True, 
                 learning_rate=1e-3, 
                 bn_decay=0.9,
                 **kwargs):
        """
        Conditional RNN RealNVP
        @param steps: rolling window moving steps
        @param n_features: No. features
        @param cond_size: No. time covariates
        @param num_blocks: No. Coupleing and BN layers
        @param rnn_units: hidden units of the encoder-decoder LSTM model
        @param st_units: hidden units of the st-network
        @param batch_norm: add Batch Normalization layer
        @param learning_rate
        """
        super(CondRealNVP, self).__init__(name='CondRealNVP', **kwargs)

        self.steps = steps
        self.n_features = n_features
        self.cond_size = cond_size
        self.num_blocks = num_blocks
        self.st_units = st_units
        self.bn_decay = bn_decay
        self.mask = np.array([int(i%2==0) for i in range(n_features)])

        self.batch_norm = batch_norm
        self.base_dist = tfd.Normal(loc=tf.zeros(n_features, dtype=tf.float32), scale=tf.ones(n_features, dtype=tf.float32))
        learning_rate_fn = tfk.optimizers.schedules.PolynomialDecay(initial_learning_rate=learning_rate, 
                                                                    decay_steps=50, 
                                                                    end_learning_rate=1e-6)
        self.optimizer = tf.optimizers.Adam(learning_rate_fn)
        self.loss_tracker = tfk.metrics.Mean(name="loss")
        
        if rnn is None:
            self.rnn = LSTM_ED(n_features=self.n_features, ncoder_hiddens=encoder_units, 
                               decoder_hiddens=decoder_units, steps=self.steps, cell_type=rnn_cell_type)
        else:
            self.rnn = rnn

    def build(self, input_shape):
        self.flow = []
        for i in range(self.num_blocks):
            self.flow += [CouplingLayer(i, n_features=self.n_features, mask=self.mask, st_units=self.st_units)]
            if self.batch_norm:
                self.flow += [BatchNormLayer(i, self.n_features, decay=self.bn_decay)]
            self.mask = 1 - self.mask
        super(CondRealNVP, self).build(input_shape)

    def forward(self, z, condition_input):
        x = z
        for module in self.flow[::-1]:
            x = module.forward(x, condition_input)
        return x
    
    def call(self, inputs, training=True):
        x, covars = inputs[..., :-self.cond_size], inputs[..., -self.cond_size:]
        
        context = inputs[:, :-self.steps]
        target_pred = self.rnn(context, target_covars=None, training=training)
        
        # (batch_sz, steps, n_features) --> (batch_sz*steps, n_features)
        target = tf.reshape(x[:, -self.steps:, :], (-1, self.n_features))
        target_pred = tf.reshape(target_pred, (-1, tf.shape(target_pred)[-1]))
        target_covars = tf.reshape(covars[:, -self.steps:, :], (-1, self.cond_size))
    
        # shuffle the time dimension for better generalization
        temp = tf.concat([target, target_pred, target_covars], axis=-1)
        tf.random.shuffle(temp)
        target, target_pred, target_covars = temp[:, :tf.shape(target)[-1]], temp[:, tf.shape(target)[-1]:-tf.shape(target_covars)[-1]], temp[:, -tf.shape(target_covars)[-1]:]
        
        sum_ildj, z = 0, target
        for module in self.flow:
            z, ildj = module(inputs=(z, tf.concat([target_pred, target_covars], axis=-1)), kwargs={'training': training})
            sum_ildj += ildj
        return z, sum_ildj

    def get_lose(self, x, training=True):
        z, sum_ildj = self(x, training=training)
        log_likelihood = self.base_dist.log_prob(z) + sum_ildj
        log_likelihood = tf.reduce_sum(log_likelihood, axis=-1)  # get joint log probability
        loss = -tf.reduce_mean(log_likelihood)
        return loss
    
    def get_score(self, data, joint=True):
        z, sum_ildj = self(data, training=False)
        log_likelihood = self.base_dist.log_prob(z) + sum_ildj
        if joint:
            return -tf.reduce_sum(log_likelihood, axis=-1)
        else:
            return -log_likelihood.numpy()

    def sample(self, covars, init_context, init_covars, length, n_ood_features=0, sample_range=[None, None], normal_boundary=2.):
        results = None
        l = 0
        i = 0
        context = init_context
        context_covars = init_covars

        enc_in = tf.expand_dims(tf.concat([context, context_covars], axis=-1), 0)
        target_pred = self.rnn(enc_in, target_covars=None, training=False)
        target_pred = tf.reshape(target_pred, (-1, tf.shape(target_pred)[-1]))
            
        while l < length:
            target_covars = covars[i]
            if sample_range[0] is not None or sample_range[1] is not None:
                if sample_range[0] is None:
                    sample_range[0] = -np.inf
                if sample_range[1] is None:
                    sample_range[1] = np.inf
                if n_ood_features > 0:
                    low = -normal_boundary * np.ones(self.n_features, dtype=np.float32)
                    high = normal_boundary * np.ones(self.n_features, dtype=np.float32)
                    rand_feat = np.random.randint(0, self.n_features, int(self.n_features*n_ood_features))
                    low[rand_feat] = sample_range[0]
                    high[rand_feat] = sample_range[1]
                    base_dist = tfp.distributions.TruncatedNormal(loc=tf.zeros(self.n_features, dtype=tf.float32), scale=1., low=low, high=high)
                else:
                    base_dist = tfp.distributions.TruncatedNormal(loc=tf.zeros(self.n_features, dtype=tf.float32), scale=1., low=sample_range[0], high=sample_range[1])
            else:
                base_dist = self.base_dist
            z = base_dist.sample(sample_shape=(self.steps,))
            x = self.forward(z, tf.concat([target_pred, target_covars], axis=-1))
            context = tf.concat([context, x], axis=0)[self.steps:]
            context_covars = tf.concat([context_covars, covars[i]], axis=0)[self.steps:]
            target_pred = self.rnn(tf.expand_dims(tf.concat([context, context_covars], axis=-1), 0), target_covars=None, training=False) 
            target_pred = tf.reshape(target_pred, (-1, tf.shape(target_pred)[-1]))
            l += self.steps
            i += 1
            if results is None:
                results = x
            else:
                results = tf.concat([results, x], axis=0)
        return results

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            loss = self.get_lose(x, training=True)
        gradients = tape.gradient(loss, sources=self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, x):
        loss = self.get_lose(x, training=False)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}