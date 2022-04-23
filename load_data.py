from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
import shapely.speedups
shapely.speedups.enable()
from utils import *

INRIX_PATH = '/home/ubuntu/Zhuangwei/EdgeNetData/traffic_inrix'


def inspect_clusters(cluster_method='OPTICS'):
    path = '%s/clusters/year=2019/%s/labels.csv' % (INRIX_PATH, cluster_method)
    labeles = pd.read_csv(path)
    clusters = np.unique(labeles['label'])
    info = []
    for c in clusters:
        try:
            path = '%s/clusters/year=2019/%s/%d.parquet' % (INRIX_PATH, cluster_method, c)
            df = pd.read_parquet(path, engine='fastparquet')
            info.append({
                'cid': c,
                'n_records': df.shape[0],
                'n_segments': df.shape[1]
            })
        except FileNotFoundError:
            pass
    return pd.DataFrame(info)


def batch_data(data, n_batches, shuffle=False):
    data = np.array_split(data, n_batches)
    if shuffle:
        np.random.shuffle(data)
    return data


def get_validation_data(data:np.array, validation_split:float):
    val_index = np.random.randint(len(data), size=int(len(data)*validation_split)).tolist()
    val_data = [data[i] for i in val_index]
    data = np.delete(data, val_index, axis=0)
    return data, np.array(val_data)


def get_sliding_data(data:np.array, window_size:int, steps=1):
    sliding_data = []
    i = 0
    while i < data.shape[0]:
        slice = data[i: i+window_size]
        if slice.shape[0] == window_size:
            sliding_data.append(slice)
        i += steps
    return np.stack(sliding_data)


# week of year, day of week, hour of day
parse_covariates = lambda ts: np.array([[t.week/52, t.dayofweek/6, t.hour/23, t.minute/59] for t in ts]).astype(np.float32)
# parse_covariates = lambda ts: np.array([[t.dayofweek/6, t.hour/23, t.minute/59] for t in ts]).astype(np.float32)

def flatten_window_data(data:np.array, steps:int):
    flatten_data = np.concatenate([data[0], np.reshape(data[1:, -steps:, :], (-1, data.shape[-1]))])
    return flatten_data


def load_inrix_cluster(cid, year, cluster_method='OPTICS', normalizer=None, test_size=0.33):
    cluster_data_path = '%s/clusters/year=%d/%s/%d.parquet' % (INRIX_PATH, year, cluster_method, cid)
    cluster_data = pd.read_parquet(cluster_data_path, engine='fastparquet')
    
    """
    We assume the training dataset doesn't contain anomalies
    """
    cols = cluster_data.columns
    X_train, X_test = train_test_split(cluster_data, test_size=test_size, shuffle=False)
    X_train_ts, X_test_ts = X_train.index, X_test.index
    cluster_data = cluster_data.values
    scaler = None
    if normalizer == 'SS':
        scaler = StandardScaler()
    elif normalizer == 'MM':
        scaler = MinMaxScaler()
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = pd.DataFrame(X_train, columns=cols, index=X_train_ts).astype(np.float32)
        X_test = pd.DataFrame(X_test, columns=cols, index=X_test_ts).astype(np.float32)
    return X_train, X_test, None, None


def load_map_inrix(set_index=False):
    inrix_map = gpd.read_file('%s/map-inrix-21.2/USA_Tennessee.geojson' % INRIX_PATH)
    if set_index:
        inrix_map.set_index('XDSegID', inplace=True)
    return inrix_map


def get_inrix_mesh(segments, n=100, inrix_map=None):
    """
    Args
    
    segments: filter segments from the Inrix Map
    n: divide the region into n*n grids
    """
    if inrix_map is None:
        inrix_map = load_map_inrix()
        inrix_map.set_index('XDSegID', inplace=True)
    region_map = inrix_map.loc[segments]
    linestr = region_map['geometry']
    pointList = []
    for ls in linestr:
        xy = ls.xy
        for i in range(len(xy[0])):
            pointList.append([xy[0][i], xy[1][i]])
    pointList = np.array(pointList)
    x_min = np.min(pointList[:, 0])
    x_max = np.max(pointList[:, 0])
    y_min = np.min(pointList[:, 1])
    y_max = np.max(pointList[:, 1])

    # divide lat and long to N segments
    x_segs = np.arange(x_min, x_max, (x_max-x_min)/n)
    y_segs = np.arange(y_min, y_max, (y_max-y_min)/n)
    x_segs = np.append(x_segs, x_segs[-1]+(x_max-x_min)/n)
    y_segs = np.append(y_segs, y_segs[-1]+(y_max-y_min)/n)

    polys = []
    gid = 0
    for i in range(len(x_segs)-1):
        for j in range(len(y_segs)-1):
            poly = Polygon([[x_segs[i], y_segs[j]], [x_segs[i+1], y_segs[j]], [x_segs[i+1], y_segs[j+1]], [x_segs[i], y_segs[j+1]]])
            polys.append(poly)

    # remove empty polys
    start_cords = region_map[['StartLong', 'StartLat']].values
    start_pnts = [Point(xy) for xy in start_cords]
    unempty_polys = []
    for ply in polys:
        empty = True
        for pnt in start_pnts:
            if ply.contains(pnt):
                empty = False
                break
        if not empty:
            unempty_polys.append(ply)
    return np.array(unempty_polys)


def load_alerts(cid, test_size=0.33):
    cluster_data_path = '%s/clusters/year=2019/OPTICS/alerts/%d.csv' % (INRIX_PATH, cid)
    accidents = pd.read_csv(cluster_data_path, index_col=0)
    y_train, y_test = train_test_split(accidents, test_size=test_size, shuffle=False)
    return y_train, y_test


