import sys
sys.path.insert(0, '../')
sys.path.insert(1, '../../')
import tensorflow as tf
from load_data import *
from utils import *
from tods.sk_interface.detection_algorithm.AutoRegODetector_skinterface import AutoRegODetectorSKI
from evaluation import *
from synthetic_gen import *
import warnings

warnings.filterwarnings('ignore')
np.random.seed(123456)
tf.random.set_seed(123456)

print('TF version:', tf.__version__)


cluster_method = 'OPTICS'

configs_ = [
    {
        "cid": -1,
        "year": 2019,
        "window": 72,
        "steps": 1
    },
    {
        "cid": 0,
        "year": 2019,
        "window": 72,
        "steps": 1
    },
    {
        "cid": 1,
        "year": 2019,
        "window": 72,
        "steps": 1
    },
    {
        "cid": 2,
        "year": 2019,
        "window": 72,
        "steps": 1
    },
    {
        "cid": 3,
        "year": 2019,
        "window": 72,
        "steps": 1
    },
    {
        "cid": 5,
        "year": 2019,
        "window": 72,
        "steps": 1
    },
    {
        "cid": 6,
        "year": 2019,
        "window": 72,
        "steps": 1
    },
    {
        "cid": 7,
        "year": 2019,
        "window": 72,
        "steps": 1
    }
]

def main(config_, i):
    config = dotdict(config_)
    log_dir = '../../experiments/Inrix/logs/AR/run%d/%s/%d' % (i, cluster_method, config.cid)
    model_dir = '../../experiments/Inrix/models/AR/%s/%d' % (cluster_method, config.cid)
    os.system('mkdir -p %s' % log_dir)
    os.system('mkdir -p %s' % model_dir)
    save_json(config_, '%s/config.json' % log_dir)

    X_train, X_test, _, _ = load_inrix_cluster(cid=config.cid, year=config.year, normalizer=None, cluster_method=cluster_method)
    segments = X_train.columns
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.values)

    model = AutoRegODetectorSKI(window_size=config.window, step_size=config.steps).primitives[-1]._clf
    model.fit(X_train)
    save_pickle(model, '%s/AutoRegODetector.pkl' % model_dir)
    model = load_pickle('%s/AutoRegODetector.pkl' % model_dir)

    syn_configs = load_json('../../config/syn_config.json')
    for syn_config in syn_configs:
        if not os.path.exists('%s/spat_%.2f_slice_%.2f/cluster_perf.json' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac'])):
            print(syn_config)
            save_json(syn_config, '%s/syn_config.json' % log_dir)
            os.system('mkdir -p %s/spat_%.2f_slice_%.2f' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac']))

            syn_data_dir = '../../syn_data/run%d/spat_%.2f_slice_%.2f/%d' % (i, syn_config['spat_frac'], syn_config['slice_frac'], config.cid)
            _, X_test, _, _ = load_inrix_cluster(cid=config.cid, year=config.year, cluster_method=cluster_method, normalizer=None)
            syn_X, syn_y, syn_interpret_label = generate_syn_ts(syn_data_dir, syn_config, source=X_test)

            tods_cluster_evaluate(syn_X, syn_y, scaler, model.decision_function, save_dir='%s/spat_%.2f_slice_%.2f' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac']))
            
        if syn_config['slice_frac'] == 0.05 and not os.path.exists('%s/spat_%.2f_slice_%.2f/seg_perf.json' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac'])):
            kde_segment_evaluate(syn_X, syn_interpret_label, config, syn_config, scaler, log_dir)
    
    
if __name__ == "__main__":
    for config_ in configs_:
        for run in range(1, 6):
            main(config_, run)