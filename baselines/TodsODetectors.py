import sys
import numpy as np
from detecta import detect_cusum
from pyod.models.knn import KNN
from pyod.models.vae import VAE
from pyod.models.combination import average
from pyod.utils.utility import standardizer
from tensorflow import keras

from tods.sk_interface.detection_algorithm.DeepLog_skinterface import DeepLogSKI
from tods.sk_interface.detection_algorithm.LSTMODetector_skinterface import LSTMODetectorSKI
from tods.sk_interface.detection_algorithm.Telemanom_skinterface import TelemanomSKI
from tods.sk_interface.detection_algorithm.KDiscordODetector_skinterface import KDiscordODetectorSKI
from tods.sk_interface.detection_algorithm.AutoRegODetector_skinterface import AutoRegODetectorSKI
from tods.sk_interface.detection_algorithm.MatrixProfile_skinterface import MatrixProfileSKI

from keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

sys.path.insert(1, '../')
from load_data import *
from utils import *


"""
@software{marcos_duarte_2021_4598962,
  author       = {Marcos Duarte},
  title        = {detecta: A Python module to detect events in data},
  month        = mar,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.5},
  doi          = {10.5281/zenodo.4598962},
  url          = {https://doi.org/10.5281/zenodo.4598962}
}
"""
def cusum_detector(data, 
                   threshold, 
                   drift=0.01, 
                   window_size=None, 
                   show=False):
    if window_size:
        batched_data = np.array_split(data, window_size)
        tas = []
        tais = []
        tafs = []
        amps = []
        for i, window in enumerate(batched_data):
            ta, tai, taf, amp = detect_cusum(window, threshold=threshold, drift=drift, ending=True, show=show)  
            if len(ta) > 0:
                tas.extend(list(i*window.shape[0]+ta))
                tais.extend(list(i*window.shape[0]+tai))
                tafs.extend(list(i*window.shape[0]+taf))
                amps.extend(list(i*window.shape[0]+amp))
        return tas, tais, tafs, amps
    else:
        ta, tai, taf, amp = detect_cusum(data, threshold=threshold, drift=drift, ending=True, show=show)
        return ta, tai, taf, amp


"""
@article{zhao2019pyod,
  author  = {Zhao, Yue and Nasrullah, Zain and Li, Zheng},
  title   = {PyOD: A Python Toolbox for Scalable Outlier Detection},
  journal = {Journal of Machine Learning Research},
  year    = {2019},
  volume  = {20},
  number  = {96},
  pages   = {1-7},
  url     = {http://jmlr.org/papers/v20/19-011.html}
}
"""
def knn_detector(X_train, X_test=None, num_classifier=6):
    k_list = [10*x for x in range(1, num_classifier+1)]
    test_scores =  None
    if X_test is not None:
        test_scores = np.zeros([X_test.shape[0], num_classifier])
    train_scores = np.zeros([X_train.shape[0], num_classifier])
    for i, k in enumerate(k_list):
        clf = KNN(n_neighbors=k, method='largest', n_jobs=-1)
        clf.fit(X_train)
        if X_test is not None:
            score = clf.decision_function(X_test)  # outlier scores
            test_scores[:, i] = score
        train_scores[:, i] = clf.decision_scores_
    if X_test is not None:
        train_scores_norm, test_scores_norm = standardizer(train_scores, test_scores)
        return average(train_scores_norm), average(test_scores_norm)
    else:
        train_scores_norm = standardizer(train_scores, test_scores)
        return average(train_scores_norm)


def lstm_ae(input_shape, hidden_neurons, hidden_activation, dropout=0):
    inputs = Input(shape=(input_shape[1], input_shape[2]))

    # encoder
    for i in  np.arange(len(hidden_neurons)//2):
        if i == 0:
            l = LSTM(hidden_neurons[0], activation=hidden_activation, return_sequences=True, kernel_regularizer=regularizers.l2(0.00), dropout=dropout)(inputs)
        elif i == len(hidden_neurons)//2-1:
            l = LSTM(hidden_neurons[len(hidden_neurons)//2], activation=hidden_activation, return_sequences=False)(l)
        else:
            l = LSTM(hidden_neurons[i], activation=hidden_activation, return_sequences=True, dropout=dropout)(l)
    
    l = RepeatVector(input_shape[1])(l)

    # decoder
    for i in  np.arange(len(hidden_neurons)//2, len(hidden_neurons)):
        l = LSTM(hidden_neurons[i], activation=hidden_activation, return_sequences=True, dropout=dropout)(l)

    output = TimeDistributed(Dense(input_shape[2]))(l)
    model = Model(inputs=inputs, outputs=output)
    return model

def ae_detector(X_train, X_test=None, 
                hidden_neurons=[64, 32, 32, 64],
                window_size=50, 
                step_size=10,
                batch_size=32,
                hidden_activation='relu',
                validation_size=0.1,
                epochs=30):
    
    sliding_X_train = get_sliding_data(X_train, window_size, step_size)
    input_shape = sliding_X_train.shape + sliding_X_train[0].shape

    model = lstm_ae(input_shape, hidden_neurons, hidden_activation)
    model.compile(optimizer='adam', loss='mae')
    print(model.summary())
    
    es = keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    history = model.fit(sliding_X_train, sliding_X_train, epochs=epochs, batch_size=batch_size, validation_split=validation_size, shuffle=False, callbacks=[es]).history
    save_pickle(history, 'ae-history.pkl')

    sliding_X_test = get_sliding_data(X_test, window_size, step_size)

    X_train_pred = model.predict(sliding_X_train)
    train_scores = np.mean(np.abs(sliding_X_train - X_train_pred), axis=-1)
    train_scores = train_scores[:, :step_size].ravel()
    train_scores = np.concatenate([train_scores, np.zeros(X_train.shape[0]-train_scores.shape[0])], axis=0)

    if X_test is not None:
        X_test_pred = model.predict(sliding_X_test)
        test_scores = np.mean(np.abs(sliding_X_test - X_test_pred), axis=-1)
        test_scores = test_scores[:, :step_size].ravel()
        test_scores = np.concatenate([test_scores, np.zeros(X_test.shape[0]-test_scores.shape[0])], axis=0)
        return train_scores, test_scores
    else:
        return train_scores


def vae_detector(X_train, X_test=None, 
                hidden_neurons=[64, 32, 32, 64],
                batch_size=32,
                hidden_activation='relu',
                output_activation='sigmoid',
                validation_size=0.1,
                epochs=30):
    
    i = hidden_neurons.index(min(hidden_neurons))
    encoder_neurons = hidden_neurons[:i]
    decoder_neurons = hidden_neurons[i+1:]
    assert len(encoder_neurons) == len(decoder_neurons)
    latent_dim = hidden_neurons[i]
    clf = VAE(encoder_neurons=encoder_neurons, 
              decoder_neurons=decoder_neurons, 
              latent_dim=latent_dim,
              batch_size=batch_size,
              hidden_activation=hidden_activation, 
              output_activation=output_activation, 
              validation_size=validation_size,
              epochs=epochs)
    clf.fit(X_train)
    train_scores = clf.decision_scores_
    if X_test is not None:
        test_scores = clf.decision_function(X_test)
        return train_scores, test_scores
    else:
        return train_scores


"""
@article{Lai_Zha_Wang_Xu_Zhao_Kumar_Chen_Zumkhawaka_Wan_Martinez_Hu_2021, 
	title={TODS: An Automated Time Series Outlier Detection System}, 
	volume={35}, 
	number={18}, 
	journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
	author={Lai, Kwei-Herng and Zha, Daochen and Wang, Guanchu and Xu, Junjie and Zhao, Yue and Kumar, Devesh and Chen, Yile and Zumkhawaka, Purav and Wan, Minyang and Martinez, Diego and Hu, Xia}, 
	year={2021}, month={May}, 
	pages={16060-16062} 
}
"""
def autoregressive_detector(X_train, X_test=None, window_size=288, step_size=12):
    clf = AutoRegODetectorSKI(window_size=window_size, step_size=step_size).primitives[-1]._clf
    clf.fit(X_train)
    training_scores = clf.decision_scores_[window_size:]
    if X_test is not None:
        testing_scores = clf.decision_function(X_test)[0][window_size:]
        return training_scores, testing_scores
    else:
        return training_scores


def matrixprofile_detector(X_train, X_test=None, window_size=288, step_size=12):
    clf = MatrixProfileSKI(window_size=window_size, step_size=step_size).primitives[-1]._clf
    clf.fit(X_train)
    training_scores = clf.decision_scores_[window_size:]
    if X_test is not None:
        testing_scores = clf.decision_function(X_test)[0][window_size:]
        return training_scores, testing_scores
    else:
        return training_scores


"""
LSTM Detector is not used as the window size and step size parameters are never used by 
"""
def lstm_detector(X_train, X_test=None,
                  window_size=288,
                  step_size=12,
                  hidden_dim=2, 
                  activation='adam',
                  batch_size=32, 
                  epochs=30, 
                  n_hidden_layer=4):
    clf = LSTMODetectorSKI(window_size=window_size, step_size=step_size, feature_dim=X_train.shape[1], hidden_dim=4, \
                           batch_size=batch_size, epochs=epochs, activation=activation, n_hidden_layer=4).primitives[-1]._clf
    clf.fit(X_train)
    training_scores = clf.decision_scores_[window_size:]
    if X_test is not None:
        testing_scores = clf.decision_function(X_test)[0][window_size:]
        return training_scores, testing_scores
    else:
        return training_scores

def deeplog_detector(X_train, X_test=None,
                     window_size=288,
                     step_size=12,
                     hidden_dim=2,  
                     batch_size=32, 
                     epochs=30, 
                     n_hidden_layer=2):
    clf = DeepLogSKI(window_size=window_size, step_size=step_size, features=X_train.shape[1], validation_size=0.1, hidden_size=hidden_dim, \
                     verbose=1, batch_size=batch_size, epochs=epochs, stacked_layers=n_hidden_layer, l2_regularizer=1e-4).primitives[-1]._clf
    clf.fit(X_train)
    training_scores = clf.decision_scores_[window_size:]
    if X_test is not None:
        testing_scores = clf.decision_function(X_test)[window_size:]
        return training_scores, testing_scores
    else:
        return training_scores


def telemanom_detector(X_train, X_test=None,
                       window_size=288,
                       step_size=1,
                       epochs=30,
                       batch_size=32,
                       layers=[16, 16]):
    clf = TelemanomSKI(smoothing_perc=0.05, window_size=window_size, l_s=window_size, n_predictions=step_size, epochs=epochs, batch_size=batch_size, layers=layers, \
                       validation_split=0.1, lstm_batch_size=batch_size, dropout=0.3).primitives[-1]._clf
    clf = clf.fit(X_train)
    training_scores = clf.decision_function(X_train)[0]
    if X_test is not None:
        testing_scores = clf.decision_function(X_test)[0]
        return training_scores, testing_scores
    else:
        return training_scores

"""
KDiscord first split multivariate time series into subsequences (matrices), and it use kNN outlier detection based on PyOD.
For an observation, its distance to its kth nearest neighbor could be viewed as the outlying score. 
It could be viewed as a way to measure the density.
"""
def kdiscord_detector(X_train, X_test=None,
                      window_size=288,
                      step_size=12,
                      n_neighbors=5,
                      radius=1.0):
    clf = KDiscordODetectorSKI(window_size=window_size, step_size=step_size, n_neighbors=n_neighbors, radius=radius).primitives[-1]._clf
    clf.fit(X_train)
    training_scores = clf.decision_scores_
    if X_test is not None:
        testing_scores = clf.decision_function(X_test)[0]
        return training_scores, testing_scores
    else:
        return training_scores