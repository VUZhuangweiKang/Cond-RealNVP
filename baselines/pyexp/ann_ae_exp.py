import os
import sys
sys.path.insert(0, '../')
sys.path.insert(1, '../../')

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from load_data import *
from utils import *
from ann_ae import AE
from evaluation import *
from synthetic_gen import *
import warnings

warnings.filterwarnings('ignore')
np.random.seed(123456)
tf.random.set_seed(123456)

print('TF version:', tf.__version__)
print('TFP version:', tfp.__version__)

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


cluster_method = 'OPTICS'


configs_ = [
    {
        "cid": -1,
        "year": 2019,
        "batch_size": 64,
        "encoder_neurons": [64, 32],
        "decoder_neurons": [32, 64],
        "latent_dims": 2,
        "epochs": 300,
        "lr": 1e-4,
        "early_stop": 30
    },
    {
        "cid": 0,
        "year": 2019,
        "batch_size": 64,
        "encoder_neurons": [128, 64],
        "decoder_neurons": [64, 128],
        "latent_dims": 10,
        "epochs": 300,
        "lr": 1e-4,
        "early_stop": 30
    },
    {
        "cid": 1,
        "year": 2019,
        "batch_size": 64,
        "encoder_neurons": [256, 128],
        "decoder_neurons": [128, 256],
        "latent_dims": 20,
        "epochs": 300,
        "lr": 1e-4,
        "early_stop": 30
    },
    {
        "cid": 2,
        "year": 2019,
        "batch_size": 64,
        "encoder_neurons": [128, 64],
        "decoder_neurons": [64, 128],
        "latent_dims": 10,
        "epochs": 300,
        "lr": 1e-4,
        "early_stop": 30
    },
    {
        "cid": 3,
        "year": 2019,
        "batch_size": 64,
        "encoder_neurons": [128, 64],
        "decoder_neurons": [64, 128],
        "latent_dims": 10,
        "epochs": 300,
        "lr": 1e-4,
        "early_stop": 30
    },
    {
        "cid": 5,
        "year": 2019,
        "batch_size": 64,
        "encoder_neurons": [64, 32],
        "decoder_neurons": [32, 64],
        "latent_dims": 2,
        "epochs": 300,
        "lr": 1e-4,
        "early_stop": 30
    },
    {
        "cid": 6,
        "year": 2019,
        "batch_size": 64,
        "encoder_neurons": [64, 32],
        "decoder_neurons": [32, 64],
        "latent_dims": 2,
        "epochs": 300,
        "lr": 1e-4,
        "early_stop": 30
    },
    {
        "cid": 7,
        "year": 2019,
        "batch_size": 64,
        "encoder_neurons": [128, 64],
        "decoder_neurons": [64, 128],
        "latent_dims": 10,
        "epochs": 300,
        "lr": 1e-4,
        "early_stop": 30
    }
]


def main(config_, i):
    config = dotdict(config_)

    log_dir = '../../experiments/Inrix/logs/ANN_AE/run%d/%s/%d' % (i, cluster_method, config.cid)
    model_dir = '../../experiments/Inrix/models/ANN_AE/%s/%d' % (cluster_method, config.cid)
    os.system('mkdir -p %s' % log_dir)
    save_json(config_, '%s/config.json' % log_dir)

    X_train, X_test, _, y_test = load_inrix_cluster(cid=config.cid, year=config.year, normalizer=None, cluster_method=cluster_method)
    segments = X_train.columns
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.values)
    np.random.shuffle(X_train)

    model = AE(n_features=len(segments), 
                encoder_hiddens=config.encoder_neurons, 
                decoder_hiddens=config.decoder_neurons, 
                latent_dims = config.latent_dims,
                lr=config.lr, 
                model_dir='%s/ann_ae'%model_dir)
    model.compile()
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    filepath = '%s/model_weights' % model_dir
#     checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', save_frequency=1)

#     history = model.fit(X_train.astype(np.float32), 
#                         batch_size=config.batch_size, 
#                         epochs=config.epochs, 
#                         verbose=2, 
#                         validation_split=0.2, 
#                         callbacks=[es, checkpoint]).history
    model.load_weights('%s/model_weights' % model_dir)
    
    # 1. evaluation on the synthetic dataset
#     syn_configs = load_json('../../config/syn_config.json')
#     for syn_config in syn_configs:
#         if not os.path.exists('%s/spat_%.2f_slice_%.2f/cluster_perf.json' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac'])):
#             save_json(syn_config, '%s/syn_config.json' % log_dir)
#             os.system('mkdir -p %s/spat_%.2f_slice_%.2f' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac']))

#             syn_data_dir = '../../syn_data/run%d/spat_%.2f_slice_%.2f/%d' % (i, syn_config['spat_frac'], syn_config['slice_frac'], config.cid)
#             _, X_test, _, _ = load_inrix_cluster(cid=config.cid, year=config.year, cluster_method=cluster_method, normalizer=None)
#             syn_X, syn_y, syn_interpret_label = generate_syn_ts(syn_data_dir, syn_config, source=X_test)

#             metrics, scores, cluster_labels = cluster_evaluate(syn_X, syn_y, config, scaler, model.get_score, NF=False, rnn_based=False,
#                                                             save_dir='%s/spat_%.2f_slice_%.2f' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac']))
        
#         if syn_config['slice_frac'] == 0.05 and not os.path.exists('%s/spat_%.2f_slice_%.2f/seg_perf.json' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac'])):
#             results, pred_seg_labels = kde_segment_evaluate(syn_X, syn_interpret_label, config, syn_config, scaler, log_dir)
#             evaluate(syn_X, syn_interpret_label, pred_seg_labels, log_dir, config, syn_config)
    
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
    roc_evaluate(X_test, y_test_, config, scaler, model.get_score, save_dir=log_dir)
    

if __name__ == "__main__":
    for config_ in configs_:
        # for run in range(2, 6):
        #     main(config_, run)
        main(config_, 1)