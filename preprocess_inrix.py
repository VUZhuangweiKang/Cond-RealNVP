import pandas as pd
import dask.dataframe as dd
import numpy as np
import glob
import geopandas as gpd
import os, sys


WORK_PATH = '/home/ubuntu/EdgeNetData/'


if __name__ == "__main__":
    inrix_map = gpd.read_file('%s/map-inrix-21.2/USA_Tennessee.geojson' % WORK_PATH)
    inrix_map = inrix_map[inrix_map['FRC']=='1']
    inrix_map.set_index('XDSegID', inplace=True)
    inrix_map.sort_index(inplace=True)

    year = int(sys.argv[1])
    mi = int(sys.argv[2])
    mj = int(sys.argv[3])
    
    segments = inrix_map.index.astype(int)
    counties = np.unique(inrix_map['County'])
    for county in counties:
        for m in range(mi, mj):
            files = glob.glob('%s/traffic_inrix/year=%d/month=%d/county=%s/*.parquet' % (WORK_PATH, year, m, county.lower()), recursive=True)
            if len(files)==0: continue
            data = dd.read_parquet(files, engine='fastparquet')
            data = data.compute()
            data = data[data['xd_id'].astype(int).isin(segments)]
            for xd_id, df in data.groupby(by='xd_id'):
                if os.path.exists('%s/segments/%d/month.%d.parquet' % (WORK_PATH, int(xd_id), m)): continue
                # df = df[(df['confidence_score']>20) & (df['confidence_score']<=30)]
                df = df.assign(extreme_congestion=lambda x: (x.average_speed-x.speed)/x.reference_speed)
                df = df.assign(regular_congestion=lambda x: (x.reference_speed-x.speed)/x.reference_speed)
                df = df.sort_values(by='measurement_tstamp')
                df = df.set_index('measurement_tstamp')
                if m < 12:
                    date_rng = pd.date_range(start='%d-%d-01'%(year,m), end='%d-%d-01'%(year,m+1), freq='300S', closed='left')
                elif m==12:
                    date_rng = pd.date_range(start='%d-%d-01'%(year,m), end='%d-%d-31'%(year,m), freq='300S', closed='left')
                df = df.loc[date_rng[0]:date_rng[-1]]
                df = df.reindex(date_rng)
                print(xd_id, df.shape, df.dropna().shape, m)
                if not os.path.exists('%s/segments/%d' % (WORK_PATH, int(xd_id))):
                    os.system('mkdir -p %s/segments/%d' % (WORK_PATH, int(xd_id)))
                df.to_parquet('%s/segments/%d/month.%d.parquet' % (WORK_PATH, int(xd_id), m), engine='fastparquet')
