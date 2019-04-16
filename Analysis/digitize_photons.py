import numpy as np
import pandas as pd
from Analysis import *
from changepoint_process import *


def digitize_photonstamps(file_path_hdf5, pars=(1, 0.1, 0.9, 2),
                          bintime=5e-3, int_photon=False,
                          nano_time=False, real_countrate=False,
                          duration_cp=False):
    """bin=1 in millisecond
    foldername should be given as r'D:\Research\...'
    """
    def find_closest(A, target):
        # https://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
        #A must be sorted
        idx = A.searchsorted(target)
        idx = np.clip(idx, 1, len(A) - 1)
        left = A[idx - 1]
        right = A[idx]
        idx -= target - left < right - target
        return idx
    out = changepoint_photonhdf5(file_path_hdf5, time_sect=25, pars=pars)
    [hdf5_anal, timestamps, cp_out] = out
    # removing consecutive repeatition in countrate
    cp_out_cor = changepoint_output_corr(cp_out)
    state_diff = cp_out_cor['cp_state'].diff().values
    # replace 1st nan by '0'
    state_diff = np.append([1], state_diff[1:], axis=0)
    df = cp_out_cor[state_diff != 0].reset_index(drop=True)

    tmin = min(timestamps)
    tmax = max(timestamps)
    time_cp = df['cp_ts'].values
    df_tag = df['cp_state'].values
    countrate_cp = df['cp_countrate'].values
    # extract from hdf5 file
    h5 = h5py.File(file_path_hdf5)
    unit = h5['photon_data']['timestamps_specs']['timestamps_unit'][...]
    tcspc_unit = h5['photon_data']['nanotimes_specs']['tcspc_unit'][...]
    #replaces the old time stamps
    timestamps = unit * h5['photon_data']['timestamps'][...]
    nanotimes = 1e9 * tcspc_unit * h5['photon_data']['nanotimes'][...]
    mask = np.logical_and(timestamps >= tmin, timestamps <= tmax)
    timestamps = timestamps[mask]
    nanotimes = nanotimes[mask]
    h5.close()
    # closest_limit =
    idx_closest = find_closest(timestamps, time_cp)
    timestamps_closest = timestamps[idx_closest]
    # photonwise tag initiation
    dig_cp = np.digitize(timestamps, timestamps_closest)
    dig_uniq = np.unique(dig_cp)
    bins = int((max(timestamps) - min(timestamps)) / bintime)
    cr, t = np.histogram(timestamps, bins=bins)  # cr for real countrate
    dig_bin = np.digitize(timestamps, t[:-1])
    # put them in dataframe IMP keep the sequence as it is
    df_dig = pd.DataFrame()
    df_dig['timestamps'] = timestamps
    if nano_time:
        df_dig['nanotimes'] = nanotimes
    df_dig['dig_bin'] = dig_bin
    df_dig['dig_bin'] = (bintime * (df_dig['dig_bin'] - 1) + tmin)
    df_dig['cp_no'] = dig_cp
    df_dig['countrate_cp'] = dig_cp
    df_dig['countrate_cp'] = df_dig['countrate_cp'].replace(dig_uniq, countrate_cp)
    df_dig['state'] = dig_cp  # initiating array for state assigment
    avg_countrate = np.average(countrate_cp)  # from change point_list
    df_dig['state'][df_dig['countrate_cp'] < avg_countrate] = 1
    df_dig['state'][df_dig['countrate_cp'] > avg_countrate] = 2
    if real_countrate:
        df_dig['countrate'] = dig_bin  # initiating array for real countrate
        count = (1 / bintime) * df_dig.groupby('countrate').timestamps.count()
        dig_bin_uniq = np.unique(dig_bin)
        df_dig['countrate'] = df_dig['countrate'].replace(dig_bin_uniq, count)
    if duration_cp:
        df_dig['duration_cp'] = dig_cp
        t_left = df_dig.groupby('cp_no').timestamps.min();
        t_right = df_dig.groupby('cp_no').timestamps.max();
        duration = (t_right-t_left).values;
        df_dig['duration_cp'] = df_dig['duration_cp'].replace(dig_uniq, duration)                
    if int_photon:
        interphoton = np.diff(timestamps)
        df_dig = df_dig[1:]
        df_dig['int_photon'] = interphoton

    return df_dig
