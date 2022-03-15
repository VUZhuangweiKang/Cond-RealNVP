import sys
sys.path.insert(0, '../')
sys.path.insert(1, '../../')
import tensorflow as tf
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from load_data import *
from utils import *
from tods.sk_interface.detection_algorithm.DeepLog_skinterface import DeepLogSKI
from evaluation import *
from synthetic_gen import *
np.random.seed(123456)
tf.random.set_seed(123456)

print('TF version:', tf.__version__)

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

physical_devices = tf.config.list_physical_devices('XLA_GPU')

cluster_method = 'OPTICS'
np.random.seed(123456)
tf.random.set_seed(123456)

configs_ = [
    {
        "cid": -1,
        "year": 2019,
        "window": 72,
        "steps": 12,
        "batch_size": 256,
        "deeplog": {
            "hidden_size": 64,
            "l2_regularizer": 1e-4,
            "stacked_layers": 2,
            "epochs": 10,
            "activation": "tanh"
        }
    },
    {
        "cid": 0,
        "year": 2019,
        "window": 72,
        "steps": 12,
        "batch_size": 256,
        "deeplog": {
            "hidden_size": 256,
            "l2_regularizer": 1e-4,
            "stacked_layers": 2,
            "epochs": 10,
            "activation": "tanh"
        }
    },
    {
        "cid": 1,
        "year": 2019,
        "window": 72,
        "steps": 12,
        "batch_size": 256,
        "deeplog": {
            "hidden_size": 256,
            "l2_regularizer": 1e-4,
            "stacked_layers": 2,
            "epochs": 10,
            "activation": "tanh"
        }
    },
    {
        "cid": 2,
        "year": 2019,
        "window": 72,
        "steps": 12,
        "batch_size": 256,
        "deeplog": {
            "hidden_size": 128,
            "l2_regularizer": 1e-4,
            "stacked_layers": 2,
            "epochs": 10,
            "activation": "tanh"
        }
    },
    {
        "cid": 3,
        "year": 2019,
        "window": 72,
        "steps": 12,
        "batch_size": 256,
        "deeplog": {
            "hidden_size": 128,
            "l2_regularizer": 1e-4,
            "stacked_layers": 2,
            "epochs": 10,
            "activation": "tanh"
        }
    },
    {
        "cid": 5,
        "year": 2019,
        "window": 72,
        "steps": 12,
        "batch_size": 256,
        "deeplog": {
            "hidden_size": 64,
            "l2_regularizer": 1e-4,
            "stacked_layers": 2,
            "epochs": 10,
            "activation": "tanh"
        }
    },
    {
        "cid": 6,
        "year": 2019,
        "window": 72,
        "steps": 12,
        "batch_size": 256,
        "deeplog": {
            "hidden_size": 64,
            "l2_regularizer": 1e-4,
            "stacked_layers": 2,
            "epochs": 10,
            "activation": "tanh"
        }
    },
    {
        "cid": 7,
        "year": 2019,
        "window": 72,
        "steps": 12,
        "batch_size": 256,
        "deeplog": {
            "hidden_size": 128,
            "l2_regularizer": 1e-4,
            "stacked_layers": 2,
            "epochs": 10,
            "activation": "tanh"
        }
    }
]

def main(config_, i):
    config = dotdict(config_)
    dl_config = dotdict(config.deeplog)

    log_dir = '../../experiments/Inrix/logs/DeepLog/run%d/%s/%d' % (i, cluster_method, config.cid)
    model_dir = '../../experiments/Inrix/models/DeepLog/%s/%d' % (cluster_method, config.cid)
    os.system('mkdir -p %s' % log_dir)
    os.system('mkdir -p %s' % model_dir)
    save_json(config_, '%s/config.json' % log_dir)

    X_train, X_test, _, _ = load_inrix_cluster(cid=config.cid, year=config.year, normalizer=None, cluster_method=cluster_method)
    segments = X_train.columns
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.values)

    model = DeepLogSKI(window_size=config.window, 
                step_size=config.steps, 
                features=X_train.shape[1], 
                hidden_size=dl_config.hidden_size,
                batch_size=config.batch_size, 
                epochs=dl_config.epochs, 
                preprocessing=False,
                stacked_layers=dl_config.stacked_layers).primitives[-1]._clf
    # model.fit(X_train)
    # save_pickle(model, '%s/DeepLog.pkl' % model_dir)
    model = load_pickle('%s/DeepLog.pkl' % model_dir)

    # 1. evaluation on the synthetic dataset
#     syn_configs = load_json('../../config/syn_config.json')
#     for syn_config in syn_configs:
#         save_json(syn_config, '%s/syn_config.json' % log_dir)
#         os.system('mkdir -p %s/spat_%.2f_slice_%.2f' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac']))
#         syn_data_dir = '../../syn_data/run%d/spat_%.2f_slice_%.2f/%d' % (i, syn_config['spat_frac'], syn_config['slice_frac'], config.cid)
#         _, X_test, _, _ = load_inrix_cluster(cid=config.cid, year=config.year, cluster_method=cluster_method, normalizer=None)
#         syn_X, syn_y, syn_interpret_label = generate_syn_ts(syn_data_dir, syn_config, source=X_test)

#         if not os.path.exists('%s/spat_%.2f_slice_%.2f/cluster_perf.json' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac'])):
#             metrics, scores, cluster_labels = tods_cluster_evaluate(syn_X, syn_y, scaler, model.decision_function,
#                                             save_dir='%s/spat_%.2f_slice_%.2f' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac']))
            
#         if syn_config['slice_frac'] == 0.05 and not os.path.exists('%s/spat_%.2f_slice_%.2f/seg_perf.json' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac'])):
#             results, segment_labels = kde_segment_evaluate(syn_X, syn_interpret_label, config, syn_config, scaler, log_dir)
#             evaluate(syn_X, syn_interpret_label, segment_labels, log_dir, config, syn_config)

    # 2. evaluation on the accident dataset
    os.system('mkdir -p %s/alerts' % log_dir)
    _, y_test = load_alerts(cid=config.cid)
    X_test.index = pd.to_datetime(X_test.index)
    y_test.index = pd.to_datetime(y_test.index)
    X_test = X_test.loc['2019-11-15 00:00:00':'2020-01-01 00:00:00']
    y_test = y_test.loc['2019-11-15 00:00:00':'2020-01-01 00:00:00']
    
    y_test_ = np.reshape((np.sum(y_test, axis=1)>0).values, (-1, 1))
    print(config.cid, np.sum(y_test_), np.sum(y_test_)/y_test_.shape[0])
    # metrics, scores, cluster_labels = cluster_evaluate(X_test, y_test_, config, scaler, model.get_score, NF=False, rnn_based=False, save_dir='%s/accidents' % log_dir, best='f1')
    # print(metrics)
    # results, segment_labels = kde_accident_evaluate(X_test, y_test, config, scaler, log_dir)
    # metrics = evaluate(X_test, y_test, segment_labels, log_dir, config, NF=True)
    # print(results)
    roc_evaluate(X_test, y_test_, config, scaler, model.decision_function, save_dir=log_dir)

    
if __name__ == "__main__":
    for config_ in configs_:
        # for run in range(2, 6):
        #     main(config_, run)
        main(config_, 1)