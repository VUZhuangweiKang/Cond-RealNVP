import os, sys
from CondRealNVP import *
from load_data import *
from utils import *
from evaluation import *
from synthetic_gen import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')
tf.random.set_seed(0)

print('TF version:', tf.__version__)
print('TFP version:', tfp.__version__)

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

np.random.seed(123456)
tf.random.set_seed(123456)


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


def gen_clf_data(config, model, scaler):
    _, X_test, _, _ = load_inrix_cluster(cid=config.cid, year=config.year, cluster_method='OPTICS', normalizer=None)
    timesteps = X_test.index
    groups = np.array_split(timesteps, timesteps.shape[0]//config.steps)

    sample_X = []
    sample_y = []
    l = 288*30
    k = l//config.steps
    i = np.random.randint(config.window//config.steps, len(groups)-k)
    sample_ts = np.reshape(groups[i:i+k], (1, -1))[0]
    covars = np.apply_along_axis(lambda x: parse_covariates(pd.to_datetime(x)), 1, groups[i:i+k])

    warm_ts = groups[i-config.window//config.steps+1:i]
    warm_covars = np.apply_along_axis(lambda x: parse_covariates(pd.to_datetime(x)), 1, warm_ts).reshape(config.window-config.steps, covars.shape[-1])
    warm_x = scaler.transform(X_test.loc[np.ravel(warm_ts)].values)

    lower_ood = model.sample(covars, warm_x, warm_covars, l, sample_range=[-2.0, -1], n_ood_features=0.5, normal_boundary=1).numpy()
    upper_ood = model.sample(covars, warm_x, warm_covars, l, sample_range=[1, 2.0], n_ood_features=0.5, normal_boundary=1).numpy()
    normals = model.sample(covars, warm_x, warm_covars, l, sample_range=[-1, 1], n_ood_features=0).numpy()
    idx = np.arange(l)
    np.random.shuffle(idx)
    lower_idx, upper_idx, normal_idx = idx[:l//4], idx[l//4:l//2], idx[l//2:]

    samples = np.zeros((l, normals.shape[-1]))
    samples[lower_idx] = lower_ood[lower_idx]
    samples[upper_idx] = upper_ood[upper_idx]
    samples[normal_idx] = normals[normal_idx]

    labels = np.ones(l)
    labels[normal_idx] = 0

    sample_X.append(samples)
    sample_y.append(labels)

    sample_X = np.reshape(sample_X, (-1, sample_X[0].shape[-1]))
    sample_y = np.ravel(sample_y)

    return sample_X, sample_y, sample_ts


if __name__ == "__main__":
    config_id = int(sys.argv[1])
    config_ = load_json('configs.json')[config_id]

    config_ = {**config_, **exp_config}
    config = dotdict(config_)

    for k in range(5):
        exp_config = dotdict(exp_config)
        log_dir = 'experiments/Inrix/logs/CondRealNVP/classifier/run%d/%d' % (k, config.cid)
        model_dir = 'experiments/Inrix/models/CondRealNVP/OPTICS/%d' % (config.cid)
        os.system('mkdir -p %s' % log_dir)
        save_json(config_, '%s/config.json' % log_dir)

        X_train, X_test, _, _ = load_inrix_cluster(cid=config.cid, year=config.year, cluster_method='OPTICS', normalizer=None)
        segments = X_train.columns
        train_idx = X_train.index 
        test_idx = X_test.index

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train.values)

        # get embending time covariate based on time index
        covars = parse_covariates(train_idx)
        n_covars = covars.shape[-1]

        X_train = np.concatenate([X_train, covars], axis=1)
        X_train = get_sliding_data(X_train, exp_config.window, exp_config.steps)

        # 1. load trained CondRealNVP model
        model = CondRealNVP(steps=exp_config.steps,
                        n_features=len(segments),
                        rnn=None,
                        rnn_cell_type='LSTM',
                        bidirectional=False,
                        attention=False,
                        cond_size=n_covars,
                        num_blocks=config.num_blocks,
                        encoder_units=config.encoder_units,
                        decoder_units=config.decoder_units,
                        teacher_fc_ratio=0.,
                        st_units=config.st_units,
                        batch_norm=True,
                        bn_decay=exp_config.bn_decay,
                        learning_rate=exp_config.lr)
        model.build(input_shape=(exp_config.batch_size, exp_config.window, X_train.shape[-1]))
        model.compile()
        filepath = '%s/model_weights' % model_dir
        model.load_weights(filepath)
        
        if not os.path.exists('classifier_data/%d/sample_X.npy' % config.cid):
            print('Generating samples...')
            os.system('mkdir -p classifier_data/%d' % config.cid)
            sample_X, sample_y, sample_ts = gen_clf_data(config, model, scaler)
            np.save('classifier_data/%d/sample_X.npy' % config.cid, sample_X)
            np.save('classifier_data/%d/sample_y.npy' % config.cid, sample_y)
            np.save('classifier_data/%d/sample_ts.npy' % config.cid, sample_ts)
        else:
            sample_X = np.load('classifier_data/%d/sample_X.npy' % config.cid)
            sample_y = np.load('classifier_data/%d/sample_y.npy' % config.cid)
            sample_ts = np.load('classifier_data/%d/sample_ts.npy' % config.cid)
        
        sample_covars = parse_covariates(pd.to_datetime(sample_ts))
        sample_X = np.reshape(sample_X, (-1, sample_X.shape[-1]))

        # 2. create MLPClassifier model
        clf = Sequential()
        clf.add(Dense(100, activation='relu'))
        clf.add(Dense(50, activation='relu'))
        clf.add(Dense(1, activation='sigmoid'))

        learning_rate_fn = tfk.optimizers.schedules.PolynomialDecay(initial_learning_rate=1e-4, decay_steps=300, end_learning_rate=1e-6)
        opt = Adam(learning_rate=1e-4)
        clf.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=300)
        checkpoint = ModelCheckpoint('%s/classifier'%model_dir, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max', save_frequency=1)
        history = clf.fit(sample_X, sample_y, epochs=1000, batch_size=64, validation_split=0.3, callbacks=[es, checkpoint])
        clf = tf.keras.models.load_model('%s/classifier' % model_dir)
        
        # 3. evaluation on synthetic data
        syn_config = load_json('config/syn_config.json')[0]
        all_auc = []
        for i in range(1, 6):
            os.system('mkdir -p %s/syn%d' % (log_dir, i))
            syn_data_dir = 'syn_data/run%d/spat_%.2f_slice_%.2f/%d' % (i, syn_config['spat_frac'], syn_config['slice_frac'], config.cid)
            _, X_test, _, _ = load_inrix_cluster(cid=config.cid, year=exp_config.year, cluster_method='OPTICS', normalizer=None)
            syn_X, syn_y, syn_interpret_label = generate_syn_ts(syn_data_dir, syn_config, source=X_test)

            syn_X_ = syn_X.loc[sample_ts]
            syn_y_ = syn_interpret_label.loc[sample_ts]

            covars = parse_covariates(syn_X_.index)
            syn_X_ = scaler.transform(syn_X_)
            syn_y_ = np.sum(syn_y_, axis=1)>0
            syn_y_pred = clf.predict(syn_X_)
            
            fpr, tpr, thresholds = metrics.roc_curve(syn_y_, syn_y_pred)
            auc = metrics.auc(fpr, tpr)
            all_auc.append(auc)
            precision, recall, thresholds = metrics.precision_recall_curve(syn_y_, syn_y_pred)
            
            np.save('%s/syn%d/fpr.npy' % (log_dir, i), fpr)
            np.save('%s/syn%d/tpr.npy' % (log_dir, i), tpr)
            np.save('%s/syn%d/precision.npy' % (log_dir, i), precision)
            np.save('%s/syn%d/recall.npy' % (log_dir, i), recall)
            np.save('%s/syn%d/thresholds.npy' % (log_dir, i), thresholds)
        np.save('%s/auc.npy' % (log_dir), np.array(all_auc))
