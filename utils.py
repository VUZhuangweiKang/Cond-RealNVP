import os
import datetime as dt
import pandas as pd
import numpy as np
import dill as pickle
import json
from tensorflow import keras
from tsmoothie.smoother import SpectralSmoother
from sklearn.metrics import confusion_matrix, auc, roc_curve
from load_data import *


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def dict_to_str(dict):
    result = ''
    for key in dict:
        result += '%s: %s ' % (key, str(dict[key]))
    return result


def minibatch_slices_iterator(length, batch_size, ignore_incomplete_batch=False):
    """
    Iterate through all the mini-batch slices.
    Args:
        length (int): Total length of data in an epoch.
        batch_size (int): Size of each mini-batch.
        ignore_incomplete_batch (bool): If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of items.
            (default :obj:`False`)
    Yields
        slice: Slices of each mini-batch.  The last mini-batch may contain
               less indices than `batch_size`.
    """
    start = 0
    stop1 = (length // batch_size) * batch_size
    while start < stop1:
        yield slice(start, start + batch_size, 1)
        start += batch_size
    if not ignore_incomplete_batch and start < length:
        yield slice(start, length, 1)


class BatchSlidingWindow(object):
    """
    Class for obtaining mini-batch iterators of sliding windows.
    Each mini-batch will have `batch_size` windows.  If the final batch
    contains less than `batch_size` windows, it will be discarded if
    `ignore_incomplete_batch` is :obj:`True`.
    Args:
        array_size (int): Size of the arrays to be iterated.
        window_size (int): The size of the windows.
        batch_size (int): Size of each mini-batch.
        excludes (np.ndarray): 1-D `bool` array, indicators of whether
            or not to totally exclude a point.  If a point is excluded,
            any window which contains that point is excluded.
            (default :obj:`None`, no point is totally excluded)
        shuffle (bool): If :obj:`True`, the windows will be iterated in
            shuffled order. (default :obj:`False`)
        ignore_incomplete_batch (bool): If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of windows.
            (default :obj:`False`)
    """

    def __init__(self, array_size, window_size, batch_size, excludes=None,
                 shuffle=False, ignore_incomplete_batch=False):
        # check the parameters
        if window_size < 1:
            raise ValueError('`window_size` must be at least 1')
        if array_size < window_size:
            raise ValueError('`array_size` must be at least as large as '
                             '`window_size`')
        if excludes is not None:
            excludes = np.asarray(excludes, dtype=np.bool)
            expected_shape = (array_size,)
            if excludes.shape != expected_shape:
                raise ValueError('The shape of `excludes` is expected to be '
                                 '{}, but got {}'.
                                 format(expected_shape, excludes.shape))

        # compute which points are not excluded
        if excludes is not None:
            mask = np.logical_not(excludes)
        else:
            mask = np.ones([array_size], dtype=np.bool)
        mask[: window_size - 1] = False
        where_excludes = np.where(excludes)[0]
        for k in range(1, window_size):
            also_excludes = where_excludes + k
            also_excludes = also_excludes[also_excludes < array_size]
            mask[also_excludes] = False

        # generate the indices of window endings
        indices = np.arange(array_size)[mask]
        self._indices = indices.reshape([-1, 1])

        # the offset array to generate the windows
        self._offsets = np.arange(-window_size + 1, 1)

        # memorize arguments
        self._array_size = array_size
        self._window_size = window_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._ignore_incomplete_batch = ignore_incomplete_batch

    def get_iterator(self, arrays):
        """
        Iterate through the sliding windows of each array in `arrays`.
        This method is not re-entrant, i.e., calling :meth:`get_iterator`
        would invalidate any previous obtained iterator.
        Args:
            arrays (Iterable[np.ndarray]): 1-D arrays to be iterated.
        Yields:
            tuple[np.ndarray]: The windows of arrays of each mini-batch.
        """
        # check the parameters
        arrays = tuple(np.asarray(a) for a in arrays)
        if not arrays:
            raise ValueError('`arrays` must not be empty')

        # shuffle if required
        if self._shuffle:
            np.random.shuffle(self._indices)

        # iterate through the mini-batches
        for s in minibatch_slices_iterator(
                length=len(self._indices),
                batch_size=self._batch_size,
                ignore_incomplete_batch=self._ignore_incomplete_batch):
            idx = self._indices[s] + self._offsets
            yield tuple(a[idx] if len(a.shape) == 1 else a[idx, :] for a in arrays)


def dt_to_epoch(dts):
    return (dts - dt.datetime(1970,1,1)).total_seconds()

def update_dict_keys(obj:dict, substr):
    new_obj = {}
    for k in obj:
        new_obj[substr+'-'+str(k)] = obj[k]
    return new_obj

def load_pickle(file):
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    return obj

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_json(file):
    with open(file) as f:
        obj = json.load(f)
    return obj

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)

def spectral_smoother(df:pd.DataFrame, fraction=0.05):
    columns = df.columns
    data = df.values
    smoother = SpectralSmoother(smooth_fraction=fraction, pad_len=1)
    data = smoother.smooth(data).smooth_data
    data = pd.DataFrame(data, columns=columns)
    return data


def save_keras_model(model, path):
    model.save('%s/model' % path)
    model.save_weights('%s/model.h5' % path)

def load_keras_model(path):
    model = keras.models.load_model('%s/model' % path)
    model.load_weights("%s/model.h5" % path)
    return model

# Time series Seasonal Difference functions
def difference(data, interval):
	return [data[i] - data[i - interval] for i in range(interval, len(data))]

def invert_difference(orig_data, diff_data, interval):
	return [diff_data[i-interval] + orig_data[i-interval] for i in range(interval, len(orig_data))]


# plot function in deepar
import matplotlib.pyplot as plt
def plot_forecast_probs(ts_entry, forecast_entry, plot_length):
    prediction_intervals = [80.0, 95.0]
    legend = ['observations', 'median prediction'] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which='both')
    plt.legend(legend, loc='upper left')
    plt.show()
    

def flatten_loss_1d(flow_loss, steps, metric='mean'):
    temp = []
    # 补齐时间错位
    for i in range(len(flow_loss)):
        t = np.concatenate([[0] * steps * i, flow_loss[i].tolist(), [0] * (flow_loss.shape[0]-1-i)*steps], axis=0)
        temp.append(t)
    temp = np.array(temp)
    scores = []
    for i in range(temp.shape[1]):
        loss = np.trim_zeros(temp[:, i])
        if metric == 'mean':
            scores.append(np.mean(loss))
        elif metric == '90th':
            scores.append(np.quantile(loss, 0.9))
        elif metric == '80th':
            scores.append(np.quantile(loss, 0.8))
        elif metric == 'max':
            scores.append(np.max(loss))
        else:
            scores.append(np.median(loss))
    scores = np.array(scores)
    return scores


def flatten_loss_2d(dates, flow_loss, window, steps, metric='median'):
    rolling_date = get_sliding_data(dates, window, steps)
    results = {}
    for i in range(rolling_date.shape[0]):
        for j in range(rolling_date.shape[1]):
            date = rolling_date[i][j]
            if date not in results:
                results.update({date: [flow_loss[i][j]]})
            else:
                results[date].append(flow_loss[i][j])
    for k in results.keys():
        if metric == 'median':
            results[k] = np.median(results[k], axis=0)
        elif metric == 'mean':
            results[k] = np.mean(results[k], axis=0)
        elif metric == 'max':
            results[k] = np.max(results[k], axis=0)
        elif metric == '90th':
            results[k] = np.quantile(results[k], 0.9, axis=0)
        elif metric == '80th':
            results[k] = np.quantile(results[k], 0.8, axis=0)
    results = pd.DataFrame(results).T
    return results


def box_plot(data, edge_color, fill_color, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 5))
    bp = ax.boxplot(data, patch_artist=False, showfliers=False, notch=True)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)  
    return bp
