"""
@Author: Zhuangwei Kang
Cluster road segments based on pairwise similarity 
"""


import pandas as pd
import numpy as np
import dask.dataframe as dd
import glob
from scipy.spatial import distance
from sklearn.cluster import OPTICS, KMeans
from k_means_constrained import KMeansConstrained
from sklearn.metrics import silhouette_score
from sktime.distances.elastic_cython import dtw_distance
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count, Pool
import folium
import load_data as ld
import argparse


BASE_PATH = '/home/ubuntu/EdgeNetData/traffic_inrix'
CORES = cpu_count()
np.random.seed(42)


def sample_inrix_mts(Y, M, D, segment_parquets, nDays=7):
    """
    Sample Inrix data to build segment MTS.
    @param Y: year
    @param M: month
    @param D: day
    @param segment_parquets: segment parquet files
    @param nDays: slice nDays data
    @return inrix: numpy array in shape (n_segments * n_timestamp * n_features) 
    """
    date_rng = pd.date_range(start='%d-%d-%d'%(Y,M,D), end='%d-%d-%d'%(Y,M+nDays//30,D+nDays%30), freq='300S', closed='left')
    inrix = []
    for seg_parquet in segment_parquets:
        mts = dd.read_parquet(seg_parquet, engine='fastparquet')
        mts = mts.loc[date_rng[0]:date_rng[-1]]
        cols = [x for x in mts.columns if 'average_speed' in x]
        mts = mts[cols]
        mts = mts.compute()
        mts.sort_index(inplace=True)
        mts = mts.interpolate(method='bfill').fillna(0)
        inrix.append(mts.values)
    inrix = np.array(inrix).reshape(len(inrix), inrix[0].shape[0], inrix[0].shape[1])
    return inrix


def dtw_worker(slice):
    index, data = slice
    rows, cols = len(index), data.shape[0]
    matrix = np.zeros(shape=(rows,cols))
    for i in range(rows):
        for j in range(cols):
            if i != j:
                matrix[i, j] = dtw_distance(data[index[i]], data[j])
    return matrix


def compute_dtw(data):
    """
    @param data: a bunch of timeseries datasets with shape (n_datasets, n_timestamps, n_features)
    """
    groups = np.array_split(np.arange(data.shape[0]), CORES)
    groups = zip(groups, [data]*CORES)
    with ThreadPoolExecutor(max_workers=CORES) as pool:
        dtw_matrix = pool.map(dtw_worker, groups)
    return np.concatenate(list(dtw_matrix))


def compute_distance(s1, s2, metric):
    val = []
    for i in range(s1.shape[0]):
        if metric == 'cityblock':
            d = distance.cityblock(s1[i], s2[i])
        elif metric == 'correlation':
            d = distance.correlation(s1[i], s2[i])
        elif metric == 'minkowski':
            d = distance.minkowski(s1[i], s2[i])
        else:
            d = distance.euclidean(s1[i], s2[i])
        val.append(d)
    return np.mean(val)


def merge_cluster(arg):
    segment_parquets, time_meta, initial_clusters, metric = arg
    Y, M, D, nDays = time_meta
    segments_data = sample_inrix_mts(Y, M, D, segment_parquets, nDays)
    cluster_labels = []
    labels = list(initial_clusters.keys())
    for n, seg in enumerate(segments_data):
        pw_dist = []
        for ic in initial_clusters:
            if metric == 'dtw':
                dist = dtw_distance(seg, initial_clusters[ic])
            else:
                dist = compute_distance(seg, initial_clusters[ic], metric)
            pw_dist.append(dist)
        cluster_labels.append(int(labels[np.argmin(pw_dist)]))
        if n % 100 == 0:
            print('Merged %d segments.' % n)
    return cluster_labels


def get_distance_metrics(n_mts1, n_mts2, metric='euclidean'):
    """
    @params: 3D numpy array: (n_mts * n_timestamps * n_features)
    @param metric: euclidean(default), cityblock, correlation, minkowski, DTW
    @return: 2D distance matrix: (n_mts * n_mts)
    """
    assert n_mts1.shape == n_mts2.shape
    pairewise_distance = np.zeros(shape=(n_mts1.shape[0], n_mts1.shape[0]))

    for i in range(n_mts1.shape[0]):
        for j in range(i):
            pairewise_distance[i][j] = compute_distance(n_mts1[i], n_mts2[j], metric)
            pairewise_distance[j][i] = pairewise_distance[i][j]
    return pairewise_distance


def auto_kmeans(X, min_clusters=3, max_clusters=20):
    """
    Iteratively searching the optimal K value based on the Silhouette coefficient.
    @param X: data for fitting KMeans model
    @param min_cluster: the minimum number of clusters
    @param max_clusters: the maximum number of clusters
    """
    best_model = None
    best_ss = -np.inf
    record = []
    for nc in range(min_clusters, max_clusters+1):
        # clf = KMeans(n_clusters=nc, init='k-means++', n_jobs=-1, random_state=0).fit(X)
        clf = KMeansConstrained(n_clusters=nc, size_min=5, random_state=0, n_jobs=-1).fit(X)
        ss = silhouette_score(X, labels=clf.labels_)
        record.append([nc, ss])
        if ss > best_ss:
            best_model = clf
            best_ss = ss
        print('No. clusters: %d, score: %f' % (nc, ss))
    record = np.array(record)
    with open('kmeans.npy', 'wb') as f:
        np.save(f, record)
    return best_model


def cluster_segments(Y, M, D, init_segments=100, nDays=1, metric='dtw', method='OPTICS'):
    """
    Compute pairewise distance metrics for K randomly sampled segments, then add the remaining segments to the K cluster
    @param Y: year
    @param M: month
    @param D: day
    @param init_segments: the number of segment samples(columns) for clustering
    @param nDays: the number of days (rows) for clustering
    @param metrics: TS similarity matrix
    @param method: clustering method, density-based: OPTICS, centroid-based: KMeans
    @return inrix_map: Inrix map after clustering
    """
    # step 1. sample segments
    segment_parquets = glob.glob('%s/year=%d/**' % (BASE_PATH, Y))
    seg_cluster_labels = np.zeros(len(segment_parquets))
    print('NO. Segments:', len(segment_parquets))
    sample_index = np.random.randint(0, len(segment_parquets), init_segments)
    random_segment_parquets = np.take(segment_parquets, sample_index)
    print('M: %d, D: %d' % (M, D))

    # step 2. generate initial clusters
    inrix_mts = sample_inrix_mts(Y, M, D, random_segment_parquets, nDays)
    print('Sampled Inrix MTS Shape:', inrix_mts.shape)
    if metric == 'dtw':
        distance_matrix = compute_dtw(inrix_mts)
    else:
        distance_matrix = get_distance_metrics(inrix_mts, inrix_mts)

    if method == 'KMeans':
        clf = auto_kmeans(distance_matrix)
    else:
        clf = OPTICS(min_samples=5, n_jobs=-1).fit(distance_matrix)
    labels = clf.labels_

    print('Initial clustering done')
    seg_cluster_labels[sample_index] = labels
    initial_clusters = {}
    for c in np.unique(labels):
        initial_clusters.update({str(c): np.mean(inrix_mts[np.where(labels==c)[0]], axis=0)})
    
    # step 3. iteratively merge other segments into known clusters
    print('Merging other segments...')
    osg_index = np.delete(np.arange(len(segment_parquets)), sample_index)
    osg_index_group = np.array_split(osg_index, CORES)
    osg_parquet_groups = [np.take(segment_parquets, x) for x in osg_index_group]
    osg_groups_args = list(zip(osg_parquet_groups, [(Y, M, D, 1)]*CORES, [initial_clusters]*CORES, [metric]*CORES))
    with ThreadPoolExecutor(max_workers=CORES) as pool:
        results = pool.map(merge_cluster, osg_groups_args)
    results = list(results)
    for i in range(CORES):
        np.put(seg_cluster_labels, osg_index_group[i], results[i])

    label_df = pd.DataFrame([])
    label_df['label'] = seg_cluster_labels
    label_df['xdsegid'] = [int(x.split('/')[-1].split('.')[0]) for x in segment_parquets]
    label_df.to_csv('%s/clusters/year=%d/%s/labels.csv' % (BASE_PATH, Y, method), index=None)
    return label_df


def generate_cluster_map(Y, method='OPTICS'):
    # step 4. generate cluster map
    inrix_map = ld.load_map_inrix()
    inrix_map.set_index('XDSegID', inplace=True)
    label_df = pd.read_csv('%s/clusters/year=%d/%s/labels.csv' % (BASE_PATH, Y, method))
    label_df.set_index('xdsegid', inplace=True)
    label_df.sort_index(inplace=True)
    inrix_map = inrix_map.join(label_df)
    inrix_map = inrix_map[inrix_map['label'].notna()]
    inrix_map = inrix_map.astype({'label': 'int64'})
    inrix_map.reset_index(inplace=True)

    all_colors = pd.read_csv('colors.csv', header=None).to_numpy().squeeze()
    all_colors = np.array_split(all_colors, label_df['label'].unique().shape[0])

    fmap = folium.Map(location = [36,-86], zoom_start=10)
    folium.GeoJson(inrix_map,
                name='Inrix',
                tooltip=folium.features.GeoJsonTooltip(fields=['XDSegID', 'PreviousXD', 'NextXDSegI', 'County', 'label'], localize=True),
                style_function=lambda seg: {'color':  all_colors[seg['properties']['label']][0], "weight": 5}).add_to(fmap)
    fmap.save('%s/clusters/year=%d/%s/InrixCluster.html' % (BASE_PATH, Y, method))
    return inrix_map


def fill_missing(args):
    """
    Fill missing values in MTS
    @args: cluster label, inrix_map, year, segment
    return: updated cluster data
    """
    label_df, inrix_map, year, col = args
    cluster_data = None
    for index, row in label_df.iterrows():
        path = '%s/year=%d/%d' % (BASE_PATH, year, index)
        data = pd.read_parquet(path, engine='fastparquet')
        if col in data.columns:
            data = data[col].to_frame()
        else:
            print(data.columns)
            continue
        data.columns = [str(index)]
        
        # 1. fill missing values using neighboring segments
        nbrs = []
        if not np.isnan(inrix_map.loc[index]['PreviousXD']):
            prevseg_path = '%s/year=%d/%d' % (BASE_PATH, year, int(inrix_map.loc[index]['PreviousXD']))
            nbrs.append(prevseg_path)
        if not np.isnan(inrix_map.loc[index]['NextXDSegI']):
            nextseg_path = '%s/year=%d/%d' % (BASE_PATH, year, int(inrix_map.loc[index]['NextXDSegI']))
            nbrs.append(nextseg_path)
        if inrix_map.loc[index]['PreviousXD'] in inrix_map.index and not np.isnan(inrix_map.loc[inrix_map.loc[index]['PreviousXD']]['PreviousXD']):
            prev2seg_path = '%s/year=%d/%d' % (BASE_PATH, year, int(inrix_map.loc[inrix_map.loc[index]['PreviousXD']]['PreviousXD']))
            nbrs.append(prev2seg_path)
        if inrix_map.loc[index]['NextXDSegI'] in inrix_map.index and not np.isnan(inrix_map.loc[inrix_map.loc[index]['NextXDSegI']]['NextXDSegI']):
            next2seg_path = '%s/year=%d/%d' % (BASE_PATH, year, int(inrix_map.loc[inrix_map.loc[index]['NextXDSegI']]['NextXDSegI']))
            nbrs.append(next2seg_path)

        for neighbor in nbrs:
            miss = data.isna().any(axis=1).index
            if len(miss)==0: break
            try:
                nb_df = pd.read_parquet(neighbor, engine='fastparquet')[col].to_frame()
                nb_df.sort_index(inplace=True)
                nb_df = nb_df.loc[miss]
                nb_df.dropna(inplace=True)
                data.loc[miss] = data.loc[miss].fillna(nb_df)
            except FileNotFoundError:
                continue
    
        # 2. fill missing values using history data
        for hy in np.arange(year-1, 2016, -1):
            miss = data.isna().any(axis=1).index
            if len(miss)==0: break
            hist_path = '%s/year=%d/%d' % (BASE_PATH, hy, index)
            hist_data = pd.read_parquet(hist_path, engine='fastparquet')[col].to_frame()
            hist_data.sort_index(inplace=True)

            hist_miss = [ts.replace(year=hy) for ts in miss]
            hist_miss = np.intersect1d(hist_miss, hist_data.index.tolist())
            temp = [ts.replace(year=year) for ts in hist_miss]

            hist_miss_data = hist_data.loc[hist_miss]
            hist_miss_data.index = data.loc[temp].index
            hist_miss_data.dropna(inplace=True)
            data.loc[miss] = data.loc[miss].fillna(hist_miss_data)
    
        if cluster_data is None:
            cluster_data = data
        else:
            cluster_data = cluster_data.join(data)
    return cluster_data


def generate_cluster(cid, year=2019, counties=['DAVIDSON'], col='extreme_congestion', method='OPTICS'):
    """
    Generate cluster data for the specified cluster.
    """
    inrix_map = ld.load_map_inrix()
    label_df = pd.read_csv('%s/clusters/year=%d/%s/labels.csv' % (BASE_PATH, year, method))
    label_df = label_df[label_df['label']==cid].astype(np.int)

    inrix_map = inrix_map[inrix_map['XDSegID'].isin(label_df['xdsegid'])]
    if counties is not None:
        inrix_map = inrix_map[inrix_map['County'].isin(counties)]
        label_df = label_df[label_df['xdsegid'].isin(inrix_map['XDSegID'])]

    inrix_map.set_index('XDSegID', inplace=True)
    label_df.set_index('xdsegid', inplace=True)
    
    label_df = np.array_split(label_df, CORES)
    arg_group = []
    for i in range(CORES):
        arg_group.append([label_df[i], inrix_map, year, col])
    with Pool(CORES) as pool:
        results = pool.map(fill_missing, arg_group)
    results = list(results)

    cluster_data = results[0]
    for result in results[1:]:
        if result is not None: # the result could be None since the specified counties may do not have the given cluster
            cluster_data = cluster_data.join(result)
    
    # interpolate missing data using nearest timestamp
    if cluster_data is None:
        return
    cluster_data = cluster_data.interpolate(method='bfill').dropna()
    cluster_data.sort_index(inplace=True)
    print('Cluster Data Shape:', cluster_data.shape)
    
    cluster_data.to_parquet('%s/clusters/year=%d/%s/%d.parquet' % (BASE_PATH, year, method, cid), engine='fastparquet')
    return cluster_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', '-y', default=2019, type=int, help='year')
    parser.add_argument('--month', '-m', type=int, help='month', default=6)
    parser.add_argument('--day', '-d', type=int, help='day', default=1)
    parser.add_argument('--nSegs', default=100, type=int, help='sample N segments to build the initial segment classifier')
    parser.add_argument('--nDays', default=7, type=int, help='the number of days used for building the initial classifier')
    parser.add_argument('--distance_metric', choices=['dtw', 'euclidean', 'cityblock', 'correlation', 'minkowski'], default='dtw', help='distance metric')
    parser.add_argument('--method', help='clustering method', choices=['KMeans', 'OPTICS'], default='OPTICS')
    args = parser.parse_args()

    Y, M, D = args.year, args.month, args.day
    labels = cluster_segments(Y, M, D, init_segments=args.nSegs, nDays=args.nDays, metric=args.distance_metric, method=args.method)
    generate_cluster_map(Y, method=args.method)
    
    labels = pd.read_csv('%s/clusters/year=%d/%s/labels.csv' % (BASE_PATH, Y, args.method))
    for lb in np.unique(labels['label']):
        generate_cluster(cid=lb, year=Y, method=args.method)