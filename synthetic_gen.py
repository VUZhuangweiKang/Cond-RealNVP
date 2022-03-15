import os
import numpy as np
from load_data import *


"""
Generate synthetic Inrix testing dataset for evalution
"""
class SyntheticTS:
    """
    Base synthetic timeseries class
    """

    def __init__(self, mts:pd.DataFrame):
        """
        Initialize

        Args:
            mts: the source multivariate timeseries
        Returns: None
        """
        self.mts = mts
        self.cols = self.mts.columns
        
        self.inrix_map = load_map_inrix()
        self.inrix_map.set_index('XDSegID', inplace=True)
        self.inrix_map = self.inrix_map.loc[[int(c) for c in self.cols]]
    
    def normal_process(self, rule=1):
        """
        Generate a clean(NO ANOMALIES) MTS based on the raw input
        
        Args:
        rule: resample rule
        dir_path: parquet file path of old resampled MTS, set to None for new operation
        """
        self.resample_rule = rule
        re_mts = self.mts.resample(rule='%dH' % rule)
        re_mts_mean = re_mts.mean()
        self.re_date_rng = re_mts_mean.index
        normal_mts = []
        for i in range(len(self.re_date_rng)-1):
            mts_slice = self.mts.loc[self.re_date_rng[i]: self.re_date_rng[i+1]]
            if i != len(self.re_date_rng)-1:
                mts_slice = mts_slice[:-1] # prevent deuplicating the first record
            cov = mts_slice.cov()  # get cov matrix
            normal_mts_slice = np.random.multivariate_normal(mean=re_mts_mean.iloc[i], cov=cov, size=mts_slice.shape[0])
            temp = mts_slice.values
            low = np.mean(temp, axis=0)-1*np.std(temp, axis=0)
            high = np.mean(temp, axis=0)+1*np.std(temp, axis=0)
            normal_mts_slice = np.clip(normal_mts_slice, low, high)
            normal_mts.extend(normal_mts_slice)
        date_rng = pd.date_range(start=self.re_date_rng[0], end=self.re_date_rng[-1], freq='300S')[:len(normal_mts)]
        self.normal_mts = pd.DataFrame(normal_mts, index=date_rng, columns=[str(c) for c in self.cols])

        self._time_slices_ = np.array_split(date_rng, len(self.re_date_rng))
        self.ano_labels = pd.DataFrame(np.zeros((self.normal_mts.shape[0], 1)), columns=['label'], index=date_rng)
        self.interpret_ano_labels = []
    
    def get_segs(self, poly:Polygon):
        """Get segments in a polygon"""
        results = []
        for xd_id, row in self.inrix_map.iterrows():
            start_pnt = Point([row['StartLong'], row['StartLat']])
            if poly.contains(start_pnt):
                results.append(xd_id)
        return np.array(results)
    
    def get_gids(self, mesh, points):   
        """
        Get grid ID of a list of points in a mesh

        Args:
        mesh ([Polygon]): List of Polygon Objects
        points ([(lon, lat)]): list of coordinate of target points
        """
        gids = []
        points = [Point(xy) for xy in points]
        for pnt in points:
            flag = False
            for i in range(len(mesh)):
                if mesh[i].contains(pnt):
                    gids.append(i)
                    flag = True
                    break
            if not flag:
                gids.append(None)
        return gids

    def anomalize(self, slice_frac=0.02, magnitude=1.0, spat_frac=0.1, subseq_frac=0.5, types=[1, 2]):
        """
        Induces anomalies in the Gaussian process

        Args:
        anomaly_frac (float): Fraction of time slices to pollute
        magnitude (float): Scale factor of anomalies. The range of the normal data will be scaled by this factor to inject anomalies.
        spat_frac (float): Fraction of segments/polygon to pollute
        subseq_frac (float): Fraction of continuous subsequence in a time slice to pollute
        Any positive number is accepted but a number > 1.0 is needed to create meaningful anomalies/outliers.
        """
        self.anomalized_data = self.normal_mts.copy()
        n_polluted_slices = int(slice_frac*len(self._time_slices_))
        # clip the first and last day due to imcomplete data
        idx = np.random.choice(np.arange(len(self._time_slices_))[24//self.resample_rule:-24//self.resample_rule], n_polluted_slices, replace=False).astype(np.int64)
        pending_slices = np.take(self._time_slices_, idx)

        if types == [1]:
            n_spatial_anomalies = len(pending_slices)
        elif types == [2]:
            n_spatial_anomalies = 0
        else:
            n_spatial_anomalies = int(0.667 * len(pending_slices))
        anomalies = np.array([1] * n_spatial_anomalies + [2] * (len(pending_slices) - n_spatial_anomalies))
        np.random.shuffle(anomalies)

        # count the frequency of temporal anomaly in a day
        temporal_freq = np.zeros((self.mts.index[-1] - self.mts.index[0]).days+1)
       
        for k, ps in enumerate(pending_slices):
            mask = np.array([1] * int(spat_frac*len(self.cols)) + [0] * (len(self.cols)-int(spat_frac*len(self.cols))))
            np.random.shuffle(mask)

            """inject spatial anomalies"""
            if anomalies[k] == 1:
                ps_len = len(ps)
                ps = pd.to_datetime(ps)
                pol_data = self.normal_mts.loc[ps]
            
                day_start = ps[0].replace(hour=0, minute=0)
                day_end = ps[0].replace(hour=23, minute=59)
                arr_min = self.normal_mts.loc[day_start:day_end].min(axis=0).values
                arr_max = self.normal_mts.loc[day_start:day_end].max(axis=0).values

                pol_data = pol_data * (1-mask) + (pol_data.mean(axis=0).values + np.random.uniform(
                    low=-magnitude*(arr_max-arr_min), high=magnitude*(arr_max-arr_min), size=(ps_len, len(self.cols))))*mask
                ano_type = 1
            else:
                """inject temporal anomalies"""
                ps_len = 2*len(ps)
                ps = pd.to_datetime(ps)  
                
                """step1. get all data in this day"""
                day_start = ps[0].replace(hour=0, minute=0, second=0)
                day_end = ps[0].replace(hour=23, minute=59, second=59)
                day_data = self.mts.loc[day_start:day_end]

                day_idx = (day_start - self.mts.index[0]).days
                temporal_freq[day_idx] += 1
                s = int(temporal_freq[day_idx])
                
                """step2. group hourly mean (over segments)"""
                hour_mean = day_data.resample(rule='1H').mean()
                hour_seg_mean = hour_mean.mean(axis=1).sort_values()
                
                """step3. flip head and tail got from the prev step, note this flip 1H data"""
                sorted_idx = hour_seg_mean.index
                left = self.mts.loc[sorted_idx[s-1]: sorted_idx[s-1].replace(minute=59)] # 1hr
                right = self.mts.loc[sorted_idx[-s]: sorted_idx[-s].replace(minute=59)]
                pol_data = pd.concat([left, right], axis=0)
                pol_data_ = pol_data.copy()
                ps = pol_data.index
                
                """step4. generate temporal anomalies"""
                pol_data.loc[left.index] = right.values
                pol_data.loc[right.index] = left.values
                pol_data = pol_data_ * (1-mask) + pol_data * mask
                ano_type = 2
            
            """select a subsequence from the target polluted time slice"""
            seq_len = int(ps_len*subseq_frac)
            sub_ps_idx = [i for i in range(ps_len-seq_len)]
            sub_ps_idx = np.random.choice(sub_ps_idx, 1, replace=False)[0]
            sub_ps = ps[sub_ps_idx:sub_ps_idx+seq_len]
            self.anomalized_data.loc[sub_ps] = pol_data[sub_ps_idx:sub_ps_idx+seq_len]
            self.ano_labels.loc[sub_ps] = ano_type
            self.interpret_ano_labels.append((sub_ps, self.cols[np.where(mask==1)[0]]))
        
        int_labels = np.zeros(shape=self.anomalized_data.shape)
        int_labels = pd.DataFrame(int_labels, columns=self.anomalized_data.columns, index=self.anomalized_data.index)
        for dta in self.interpret_ano_labels:
            int_labels.at[dta[0], dta[-1]] = 1
        self.interpret_ano_labels = int_labels


# generate synthetic testing dataset
def generate_syn_ts(syn_data_dir, syn_config, source:pd.DataFrame=None, save=True):
    if not os.path.exists(syn_data_dir):
        os.system('mkdir -p %s' % syn_data_dir)
        syn_ts = SyntheticTS(source)
        syn_ts.normal_process(rule=1)
        syn_ts.anomalize(**syn_config)
        syn_X = syn_ts.anomalized_data
        syn_y = syn_ts.ano_labels.values
        if save:
            syn_X.to_parquet('%s/syn_X.parquet' % syn_data_dir)
            np.save('%s/syn_y.npy' % syn_data_dir, syn_y)
            syn_ts.interpret_ano_labels.to_parquet('%s/syn_interpret_label.parquet' % syn_data_dir)

    syn_X = pd.read_parquet('%s/syn_X.parquet' % syn_data_dir)
    syn_y = np.load('%s/syn_y.npy' % syn_data_dir)
    syn_interpret_label = pd.read_parquet('%s/syn_interpret_label.parquet' % syn_data_dir)
    return syn_X, syn_y, syn_interpret_label
