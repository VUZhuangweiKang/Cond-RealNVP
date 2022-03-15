import os
from CondRealNVP import *
from load_data import *
from utils import *
from evaluation import *
from synthetic_gen import *
from tensorflow.keras.callbacks import ModelCheckpoint
import warnings

warnings.filterwarnings('ignore')
tf.random.set_seed(0)

print('TF version:', tf.__version__)
print('TFP version:', tfp.__version__)

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=False)


cluster_method = 'OPTICS'
exp_config = {
    "year": 2019,
    "window": 84,
    "steps": 12,
    "epochs": 300,
    "lr": 1e-4,
    "batch_size": 64,
    "bn_decay": 0.9,
    "pred_len": 288*10
}
configs_ = load_json('configs.json')

def main(config_, i):
    config_ = {**config_, **exp_config}
    config = dotdict(config_)
    log_dir = 'experiments/Inrix/logs/CondRealNVP/run%d/%s/%d' % (i, cluster_method, config.cid)
    model_dir = 'experiments/Inrix/models/CondRealNVP/%s/%d' % (cluster_method, config.cid)
    os.system('mkdir -p %s' % log_dir)
    save_json(config_, '%s/config.json' % log_dir)

    X_train, X_test, _, _ = load_inrix_cluster(cid=config.cid, year=config.year, cluster_method=cluster_method, normalizer=None)
    segments = X_train.columns
    train_idx = X_train.index 

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.values)

    # get embending time covariate based on time index
    covars = parse_covariates(train_idx)
    n_covars = covars.shape[-1]

    X_train = np.concatenate([X_train, covars], axis=-1)
    X_train = get_sliding_data(X_train, config.window, config.steps)

    model = CondRealNVP(steps=config.steps,
                        n_features=len(segments),
                        rnn=None,
                        rnn_cell_type='LSTM',
                        attention=False,
                        cond_size=n_covars,
                        num_blocks=config.num_blocks,
                        encoder_units=config.encoder_units,
                        decoder_units=config.decoder_units,
                        teacher_fc_ratio=0.,
                        st_units=config.st_units,
                        batch_norm=True,
                        bn_decay=config.bn_decay,
                        learning_rate=config.lr)
    model.build(input_shape=(1, config.window, X_train.shape[-1]))
    model.compile()
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    filepath = '%s/model_weights' % model_dir
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', save_frequency=1)
    model.fit(X_train, batch_size=config.batch_size, epochs=config.epochs, 
              verbose=2, validation_split=0.2, shuffle=True, callbacks=[es, checkpoint]).history
    model.load_weights('%s/model_weights' % model_dir)

    # 1. evaluation on synthetic dataset
    syn_configs = load_json('config/syn_config.json')
    for syn_config in syn_configs:
        print(i, syn_config)
        save_json(syn_config, '%s/syn_config.json' % log_dir)
        os.system('mkdir -p %s/spat_%.2f_slice_%.2f' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac']))
        syn_data_dir = 'syn_data/run%d/spat_%.2f_slice_%.2f/%d' % (i, syn_config['spat_frac'], syn_config['slice_frac'], config.cid)
        _, X_test, _, _ = load_inrix_cluster(cid=config.cid, year=config.year, cluster_method=cluster_method, normalizer=None)
        syn_X, syn_y, syn_interpret_label = generate_syn_ts(syn_data_dir, syn_config, source=X_test)

        # anomaly detection
        if not os.path.exists('%s/spat_%.2f_slice_%.2f/cluster_perf.json' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac'])):
            cluster_evaluate(syn_X, syn_y, config, scaler, model.get_score, NF=True, rnn_based=True,
                             save_dir='%s/spat_%.2f_slice_%.2f' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac']), best='f1')
        
        # anomaly diagnosis
        if syn_config['slice_frac'] == 0.05 and not os.path.exists('%s/spat_%.2f_slice_%.2f/seg_perf.json' % (log_dir, syn_config['spat_frac'], syn_config['slice_frac'])):
            kde_segment_evaluate(syn_X, syn_interpret_label, config, syn_config, scaler, log_dir)

        # anomaly diagnosis time
        if syn_config['spat_frac'] == 0.5:
            kde_segment_evaluate(syn_X, syn_interpret_label, config, syn_config, scaler, log_dir)


if __name__ == "__main__":
    for config_ in configs_:
        main(config_, 1)