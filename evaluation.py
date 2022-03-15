from load_data import *
from sklearn.metrics import confusion_matrix, auc, roc_curve
from scipy.stats import gaussian_kde
from sklearn import metrics
import numpy as np
import time


def get_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    if precision != 0 and recall != 0:
        f1 = 2*recall*precision/(recall+precision)
    else:
        f1 = None
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    
    return  {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'auc': auc_score,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def search_opt_threshold(scores, y_true, best='f1'):
    """
    Grid search the optimal threshold for flagging anomalies on the testing dataset
    @param scores: anomaly scores (NLL)
    @param y_true: real binary classes
    @param steps: grid search steps
    @return (optimal F1 score, optimal threshold)
    """
    opt_thrd = np.quantile(scores, 0.95).astype(float)  # default threshold
    opt = -np.inf
    for thr in scores:
        y_pred = scores > thr
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            if tp+fp ==0 or tp+fn==0:
                continue
        except:
            continue
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*recall*precision/(recall+precision)
        fpr, tpr, threshold = roc_curve(y_true, y_pred)
        auc_score = auc(fpr, tpr)
        
        if best == 'f1':
            if f1 > opt:
                opt_thrd = thr
                opt = f1
        elif best == 'auc':
            if auc_score > opt:
                opt_thrd = thr
                opt = auc_score
    return opt, float(opt_thrd)


############################### Cluster-level AD
def cluster_evaluate(X, y, config, scaler, get_score, val_size=0.3, NF=False, save_dir=None, rnn_based=False, best='f1'):
    """
    @param X, syn_y: synthetic dataset
    @param config: experiment configuration
    @param val_size: ratio of validation data
    @param scaler: MinMaxScaler fitted on the realistic dataset
    @param get_score: function of getting anomaly score
    @param NL: is normalizing flow model
    @param save_dir: save evaluation results to the directory
    @return metrics, anomaly scores, optimal threshold
    """
    # spliting test and validation set
    k = int(val_size * X.shape[0])
    idx = X.index
    covars = np.reshape(parse_covariates(idx), (idx.shape[0], -1))

    X_val, X_test = X.iloc[:k].values, X.iloc[k:].values
    y_val, y_test = y[:k], y[k:]
    val_covars, test_covars = covars[:k], covars[k:]
    val_idx, test_idx = idx[:k], idx[k:]
    
    # Search Threshold on the validation set
    inputs = scaler.transform(X_val)
    if rnn_based:        
        if NF:
            temp = np.concatenate([inputs, val_covars, y_val], axis=-1)
        else:
            temp = np.concatenate([inputs, y_val], axis=-1)
        temp = get_sliding_data(temp, config.window, config.steps)
        inputs, y_val = temp[..., :-1], temp[..., -1:]
        y_val = flatten_window_data(y_val, config.steps)

    if NF:
        scores = get_score(inputs, joint=False)
        y_val = y_val[(config.window-config.steps):]
        val_idx = flatten_window_data(get_sliding_data(np.expand_dims(val_idx, -1), config.window, config.steps), config.steps).ravel()[(config.window-config.steps):]
        # np.save('%s/val_scores.npy' % save_dir, pd.DataFrame(scores, index=val_idx, columns=X.columns))
        pd.DataFrame(scores, index=pd.to_datetime(val_idx), columns=X.columns).to_csv('%s/val_scores.csv' % save_dir)
        scores = np.sum(scores, axis=-1)
    else:
        scores = get_score(inputs)
    np.save('%s/val_scores.npy' % save_dir, scores)
    
    print(scores.shape, val_idx.shape)
    opt, opt_thrd = search_opt_threshold(scores, y_true=y_val>0, best=best)
    np.save('%s/val_idx.npy' % save_dir, val_idx)
    np.save('%s/val_ood.npy' % save_dir, val_idx[np.where(y_val>0)[0]])
    np.save('%s/val_ood_pred.npy' % save_dir, val_idx[np.where(scores>opt_thrd)[0]])
    
    # Flag anomalies on the testing set
    inputs = scaler.transform(X_test)
    if rnn_based:
        if NF:
            temp = np.concatenate([inputs, test_covars, y_test], axis=-1)
        else:
            temp = np.concatenate([inputs, y_test], axis=-1)
        temp = get_sliding_data(temp, config.window, config.steps)
        inputs, y_test = temp[..., :-1], temp[..., -1:]
        y_test = flatten_window_data(y_test, config.steps)
        
    if NF:
        scores = get_score(inputs, joint=False)
        np.save('%s/test_scores.npy' % save_dir, scores)
        y_test = y_test[(config.window-config.steps):]
        test_idx = flatten_window_data(get_sliding_data(np.expand_dims(test_idx, -1), config.window, config.steps), config.steps).ravel()[(config.window-config.steps):]
        scores = np.sum(scores, axis=-1)
    else:
        scores = get_score(inputs)
    np.save('%s/t_scores.npy' % save_dir, scores)
    
    pred_test_labels = scores>opt_thrd
    np.save('%s/test_idx.npy' % save_dir, test_idx)
    np.save('%s/test_ood.npy' % save_dir, test_idx[np.where(y_test>0)[0]])
    np.save('%s/test_ood_pred.npy' % save_dir, test_idx[np.where(pred_test_labels)[0]])
    metrics = get_metrics(y_true=y_test>0, y_pred=pred_test_labels)

    metrics['opt_threshold'] = opt_thrd
    save_json(metrics, '%s/cluster_perf.json' % save_dir)
    return metrics, scores, pred_test_labels


def tods_cluster_evaluate(X, y, scaler, get_score, val_size=0.3, save_dir=None):
    """
    @param syn_X, syn_y, syn_interpret_label: synthetic dataset
    @param config: experiment configuration
    @param val_size: ratio of validation data
    @param scaler: MinMaxScaler fitted on the realistic dataset
    @param get_score: function of getting anomaly score
    @param save_dir: save evaluation results to the directory
    @return metrics, anomaly scores, optimal threshold
    """
    # spliting test and validation set
    k = int(val_size * X.shape[0])
    idx = X.index
    covars = np.reshape(parse_covariates(idx), (idx.shape[0], -1))

    X_val, X_test = X.iloc[:k].values, X.iloc[k:].values
    y_val, y_test = y[:k], y[k:]
    val_idx, test_idx = idx[:k], idx[k:]
    
    # Search Threshold on the testing set
    inputs = scaler.transform(X_val)
    scores = get_score(inputs)
    if type(scores) is tuple:
        scores = scores[0]
    
    opt_f1, opt_thrd = search_opt_threshold(scores, y_true=y_val>0)
    np.save('%s/val_scores.npy' % save_dir, scores)
    np.save('%s/val_ood.npy' % save_dir, val_idx[np.where(y_val>0)[0]])
    np.save('%s/val_ood_pred.npy' % save_dir, val_idx[np.where(scores>opt_thrd)[0]])
    
    # Flag anomalies on the testing set
    inputs = scaler.transform(X_test)
    scores = get_score(inputs)
    if type(scores) is tuple:
        scores = scores[0]

    pred_test_labels = scores>opt_thrd
    np.save('%s/test_idx.npy' % save_dir, test_idx)
    np.save('%s/test_scores.npy' % save_dir, scores)
    np.save('%s/test_ood.npy' % save_dir, test_idx[np.where(y_test>0)[0]])
    np.save('%s/test_ood_pred.npy' % save_dir, test_idx[np.where(pred_test_labels)[0]])
    metrics = get_metrics(y_true=y_test>0, y_pred=pred_test_labels)

    metrics['opt_threshold'] = opt_thrd
    save_json(metrics, '%s/cluster_perf.json' % save_dir)
    return metrics, scores, pred_test_labels


############################### Segment-level AD
def seg_kde(X_train, samples, ood_pred):
    """
    Segment-level anomaly detection using Kernel Density Estimation
    @param X_train: real traffic data
    @param X_test: synthetic testing data
    @param ood_pred: timestamps of predicted anomalies
    @param bandwidth: the bandwidth parameter of KDE algorithm
    @param neighbots: No. neighbors timesteps of target timesteps
    @return DataFrame: negative log probability of each feature
    """
    dates = pd.DataFrame([x.hour for x in X_train.index], columns=['hour'])
    dates['date'] = X_train.index
    dates = dates.groupby(by='hour')['date'].apply(list)
    print('Collected historical data...')
    
    models = {}
    for idx in ood_pred:
        if idx.hour in models:
            continue
        data = X_train.loc[dates[idx.hour]]
        
        models.update({idx.hour: []})
        for col in data.columns:
            col_data = data[col].values
            kernel = gaussian_kde(col_data)
            models[idx.hour].append(kernel)
    print('Trained KDE models...')

    results = []
    for idx in ood_pred:
        log_prob = []
        for i, col in enumerate(X_train.columns):
            target = samples.loc[idx][col]
            log_prob.append(models[idx.hour][i].logpdf(target)[0])
        results.append(log_prob)
    results = -1 * pd.DataFrame(results, columns=X_train.columns, index=ood_pred)
    return results


def kde_segment_evaluate(X_test, labels, config, syn_config, scaler, log_dir, NF=False, rnn_based=False):
    X_train, _, _, _ = load_inrix_cluster(cid=config.cid, year=config.year, cluster_method='OPTICS', normalizer=None)
    X_train = pd.DataFrame(scaler.transform(X_train.values), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test.values), columns=X_test.columns, index=X_test.index)

    val_ood_pred = np.load('%s/spat_%.2f_slice_%.2f/val_ood_pred.npy' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac']))
    test_ood_pred = np.load('%s/spat_%.2f_slice_%.2f/test_ood_pred.npy' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac']))
    ood_pred = pd.to_datetime(np.append(val_ood_pred, test_ood_pred))

    start = time.time()
    # Train KDE Model
    results = seg_kde(X_train, X_test, ood_pred)

    # split density results
    val_nll, test_nll = results.iloc[:len(val_ood_pred)], results.iloc[len(val_ood_pred):]

    # search the optimal threshold on the anomolous part of validation set for each road segment
    opt_thrds = {}
    for col in results.columns:
        true_labels = labels.loc[val_nll.index][col] > 0
        nll = val_nll[col].values 
        opt_f1, opt_thrd = search_opt_threshold(nll, true_labels)
        opt_thrds[col] = opt_thrd

    # evaluate on the anomolous part of testing set
    pred_ood_test_labels = test_nll.copy()
    for col in results.columns:
        pred_ood_test_labels[col] = (test_nll[col] > opt_thrds[col])
    avg_diag_time = (time.time() - start)/test_nll.shape[0]

    # evaluate on the whole testing set
    test_idx = np.load('%s/spat_%.2f_slice_%.2f/test_idx.npy' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac']))
    pred_test_labels = pd.DataFrame(False, index=test_idx, columns=X_test.columns)
    for col in results.columns:
        pred_test_labels.loc[test_nll.index, col] = (test_nll[col] > opt_thrds[col])
    pred_test_labels = np.reshape(pred_test_labels.values, (-1, 1))

    real_test_labels =  labels.loc[test_idx].values > 0
    real_test_labels = np.reshape(real_test_labels, (-1, 1))
    results = get_metrics(real_test_labels, pred_test_labels)
    results['avg_diag_time'] = avg_diag_time
    print(results)
    save_json(results, '%s/spat_%.2f_slice_%.2f/seg_perf.json' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac']))
    return results, pred_ood_test_labels


def roc_evaluate(X, y, config, scaler, get_score, save_dir=None, rnn_based=False):
    # spliting test and validation set
    idx = X.index
    covars = np.reshape(parse_covariates(idx), (idx.shape[0], -1))
    
    # Search Threshold on the validation set
    inputs = scaler.transform(X)
    if rnn_based:
        inputs = get_sliding_data(inputs, config.window, config.steps)
        y = get_sliding_data(y, config.window, config.steps)
        y = flatten_window_data(y, config.steps)
    scores = get_score(inputs)
    if type(scores) is tuple:
        scores = scores[0]
    
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    precision, recall, thresholds = metrics.precision_recall_curve(y, scores)
    
    np.save('%s/alerts/fpr.npy' % save_dir, fpr)
    np.save('%s/alerts/tpr.npy' % save_dir, tpr)
    np.save('%s/alerts/precision.npy' % save_dir, precision)
    np.save('%s/alerts/recall.npy' % save_dir, recall)
    np.save('%s/alerts/thresholds.npy' % save_dir, thresholds)
    return fpr, tpr, thresholds

def nf_roc_evaluate(X, y, config, scaler, get_score, save_dir=None):
    # spliting test and validation set
    idx = X.index
    covars = np.reshape(parse_covariates(idx), (idx.shape[0], -1))
    
    # Search Threshold on the validation set
    inputs = scaler.transform(X)
    temp = np.concatenate([inputs, covars, y], axis=-1)
    temp = get_sliding_data(temp, config.window, config.steps)
    inputs, y = temp[..., :-1], temp[..., -1:]
    y = flatten_window_data(y, config.steps)

    scores = get_score(inputs, joint=False)
    y = y[(config.window-config.steps):]
    idx = flatten_window_data(get_sliding_data(np.expand_dims(idx, -1), config.window, config.steps), config.steps).ravel()[(config.window-config.steps):]
    
    sum_score = np.sum(scores, axis=-1)
    best_fpr, best_tpr, best_thrds = metrics.roc_curve(y, sum_score)
    best_prec, best_recall, best_thrds = metrics.precision_recall_curve(y, sum_score)
    best_auc = metrics.auc(best_fpr, best_tpr)
    
    for i in np.arange(0.1, 1.0, 0.05):
        score = np.quantile(scores, i, axis=-1)
        fpr, tpr, thresholds = metrics.roc_curve(y, score)
        precision, recall, thresholds = metrics.precision_recall_curve(y, score)
        auc = metrics.auc(fpr, tpr)
        if auc > best_auc:
            best_fpr, best_tpr, best_thrds = fpr, tpr, thresholds
            best_prec, best_recall = precision, recall
        
    np.save('%s/alerts/fpr.npy' % save_dir, best_fpr)
    np.save('%s/alerts/tpr.npy' % save_dir, best_tpr)
    np.save('%s/alerts/precision.npy' % save_dir, best_prec)
    np.save('%s/alerts/recall.npy' % save_dir, best_recall)
    np.save('%s/alerts/thresholds.npy' % save_dir, best_thrds)
    
    return best_fpr, best_tpr, best_thrds