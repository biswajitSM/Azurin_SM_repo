import sys
import os
import time
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import scipy
import lmfit
from lmfit import Parameters, report_fit, Model

# give path for change point program (executable)
changepoint_exe = "changepoint_program/changepoint.exe"
changepoint_exe = os.path.abspath(changepoint_exe)

def changepoint_photonhdf5(file_path_hdf5, tmin=None, tmax=None,
                           time_sect=25, pars=(1, 0.1, 0.9, 2),
                           overwrite=False):
    '''
    '''
    file_path_hdf5 = os.path.abspath(file_path_hdf5)
    file_path_hdf5analysis = file_path_hdf5[:-5] + '_analysis.hdf5'
    # check if output hdf5 file exist, else create one
    if not os.path.isfile(file_path_hdf5analysis):
        h5_analysis = h5py.File(file_path_hdf5analysis, 'w')
    else:
        h5_analysis = h5py.File(file_path_hdf5analysis, 'r+')
    # check if changepoint group exist, else create one
    grp_cp = 'changepoint'
    if not '/' + grp_cp in h5_analysis.keys():
        h5_analysis.create_group(grp_cp)
    # Read and extract time stamps from photonhdf4 file
    h5 = h5py.File(file_path_hdf5)
    unit = h5['photon_data']['timestamps_specs']['timestamps_unit'].value
    timestamps = unit * h5['photon_data']['timestamps'][...]
    if not tmin:
        tmin = min(timestamps)
    if not tmax:
        tmax = max(timestamps)
    mask = np.logical_and(timestamps >= tmin, timestamps <= tmax)
    timestamps = timestamps[mask]
    h5.close()
    # check if  exist
    data_cpars = '/' + grp_cp + '/cp_' + \
        str(pars[1]) + '_' + str(pars[2]) + '_' + str(time_sect) + 's'
    if data_cpars in h5_analysis.keys() and overwrite:
        print('already exists and deleting for new analysis')
        del h5_analysis[data_cpars]
    if not data_cpars in h5_analysis.keys() or overwrite:
        changepoint_output = changepoint_exec(
            timestamps, file_path_hdf5, time_sect=time_sect, pars=pars)  # function
        h5_analysis[data_cpars] = changepoint_output
        h5_analysis[data_cpars].attrs['parameters'] = pars
        h5_analysis[data_cpars].attrs['tmin'] = tmin
        h5_analysis[data_cpars].attrs['tmax'] = tmax
        h5_analysis[data_cpars].attrs['time_sect'] = time_sect
        cp_cols = 'cp_index, cp_ts, cp_state, cp_countrate'
        h5_analysis[data_cpars].attrs['columns'] = cp_cols
        h5_analysis.flush()
    h5_analysis.close()
    h5_saved = h5py.File(file_path_hdf5analysis, 'r')
    cp_out = pd.DataFrame(h5_saved[data_cpars][:],
                            columns = ['cp_index', 'cp_ts', 'cp_state', 'cp_countrate'])
    h5_saved.close()
    return file_path_hdf5analysis, timestamps, cp_out

def changepoint_simulatedata(simulatedhdf5, time_sect=25, pars=[1, 0.1, 0.9, 2],
                             exp=True, rise=False, overwrite=False):
    h5 = h5py.File(simulatedhdf5, 'r+')
    if exp:
        grp_exp = 'exp_changepoint'
        if not '/'+grp_exp in h5.keys():
            h5.create_group(grp_exp)
        timestamps = h5['onexp_offexp']['timestamps'][...]
        grp_cpars = '/'+ grp_exp + '/cp_'+str(pars[1])+'_'+str(pars[2])+'_'+str(time_sect)+'s'
        if grp_cpars in h5.keys() and overwrite:
            print('exist')
            del h5[grp_cpars]
        if not grp_cpars in h5.keys() or overwrite:
            changepoint_output = changepoint_exec(timestamps, simulatedhdf5, 
                                              time_sect=time_sect, pars=pars)
            h5[grp_exp][grp_cpars]=changepoint_output
            h5[grp_exp][grp_cpars].attrs['parameters'] = pars
            h5[grp_exp][grp_cpars].attrs['time_sect'] = time_sect
            cp_cols = 'cp_index, cp_ts, cp_state, cp_countrate'
            h5[grp_exp][grp_cpars].attrs['columns'] = cp_cols
        h5.flush()
    if rise:
        grp_rise = 'rise_changepoint'
        if not '/'+grp_rise in h5.keys():
            print('doesnot exist')
            h5.create_group(grp_rise)
        timestamps = h5['onrise_offrise']['timestamps'][...]
        grp_cpars = '/'+ grp_rise + '/cp_'+str(pars[1])+'_'+str(pars[2])+'_'+str(time_sect)+'s'
        if grp_cpars in h5.keys() and overwrite:
            print('exist')
            del h5[grp_cpars]
        if not grp_cpars in h5.keys() or overwrite:
            changepoint_output = changepoint_exec(timestamps, simulatedhdf5, 
                                              time_sect=time_sect, pars=pars)
            h5[grp_rise][grp_cpars]=changepoint_output
            h5[grp_rise][grp_cpars].attrs['parameters'] = pars
            h5[grp_rise][grp_cpars].attrs['time_sect'] = time_sect
            cp_cols = 'cp_index, cp_ts, cp_state, cp_countrate'
            h5[grp_rise][grp_cpars].attrs['columns'] = cp_cols
        h5.flush()
    h5.close()
    h5_saved = h5py.File(simulatedhdf5, 'r')
    cp_out = pd.DataFrame(h5_saved[grp_cpars][:],
                            columns = ['cp_index', 'cp_ts', 'cp_state', 'cp_countrate'])
    h5_saved.close()    
    return simulatedhdf5, timestamps, cp_out

def changepoint_exec(timestamps, file_path_hdf5, time_sect, pars=[1, 0.1, 0.9, 2]):
    ''' Process dat file containing arrival time
    Arguments:
    timestamps: a numpy array of arrival times of photons
    file_path_hdf5: A path where temporary files created and deleted
    time_sect: length to devide the time trace to analyze by parts
    '''
    no_div = int((timestamps[-1]-timestamps[0])/time_sect)
    if no_div<1:
        no_div=1
    time_div = np.linspace(min(timestamps), max(timestamps), no_div+1)
    # time_div = list(time_div)
    changepoint_output = pd.DataFrame()
    len_timestamps = 0
    for i in range(no_div):
        t_left = time_div[i]
        t_right = time_div[i+1]
        mask = np.logical_and(timestamps>=t_left, timestamps<=t_right)
        timestamps_sect = timestamps[mask]
        timestamps_sect_corr = min(timestamps_sect)
        timestamps_sect = timestamps_sect - timestamps_sect_corr
        #saving dat file
        file_dat_ts = file_path_hdf5[:-5]+'_ts.dat'
        np.savetxt(file_dat_ts, np.c_[timestamps_sect])
        import subprocess
        pc_sys = sys.platform
        if 'linux' in pc_sys:
            args = ("wine", changepoint_exe, file_dat_ts, str(pars[0]), str(pars[1]), str(pars[2]), str(pars[3]))
            popen = subprocess.Popen(args, stdout=subprocess.PIPE)
            # popen.wait()
            output = popen.stdout.read()
        if 'win' in pc_sys:
            args = (changepoint_exe, file_dat_ts, str(pars[0]), str(pars[1]), str(pars[2]), str(pars[3]))
            popen = subprocess.Popen(args, stdout=subprocess.PIPE)
            # popen.wait()
            output = popen.stdout.read()            
        # list of output files
        file_dat_cp = file_dat_ts + '.cp'
        file_dat_em2 = file_dat_ts + '.em.2'
        file_dat_ah = file_dat_ts + '.ah'
        file_dat_bic = file_dat_ts + '.bic'
        file_dat_cp0 = file_dat_ts + '.cp0'
        #Extract proper outputs
        changepoint_cp = pd.read_csv(file_dat_cp, header=None, sep='\s+')
        cp_index_0 = pd.DataFrame([[0, 0, 0]])
        cp_index_end = pd.DataFrame([[len(timestamps_sect)-1, 0, 0]])
        changepoint_cp = pd.concat([cp_index_0, changepoint_cp, cp_index_end],
                                axis=0).reset_index(drop=True)

        changepoint_em2 = pd.read_csv(file_dat_em2, header=None, sep='\s+')
        em_add_0 = pd.DataFrame([changepoint_em2.iloc[0, :].values])
        changepoint_em2 = pd.concat([em_add_0, changepoint_em2],
                                axis=0).reset_index(drop=True)
        changepoint_ts = pd.DataFrame(timestamps_sect[changepoint_cp[0].values] + timestamps_sect_corr)# + timestamps_sect_corr
        changepoint_comb = pd.concat([changepoint_cp[[0]], changepoint_ts[[0]],
                                changepoint_em2[[0]], changepoint_em2[[1]]], axis=1)
        changepoint_comb.columns = ['cp_index', 'cp_ts', 'cp_state', 'cp_countrate']
        changepoint_comb['cp_index'] = changepoint_comb['cp_index'] + len_timestamps
        len_timestamps = len_timestamps + len(timestamps_sect)# update index
        changepoint_output = pd.concat([changepoint_output, changepoint_comb]).reset_index(drop=True)    
    # remove generated files
    os.remove(file_dat_ts)
    os.remove(file_dat_cp)
    os.remove(file_dat_em2)
    os.remove(file_dat_ah)
    os.remove(file_dat_bic)
    os.remove(file_dat_cp0)
    return changepoint_output

def changepoint_output_corr(changepoint_output):
    cp_ts = []
    for i, j in zip(changepoint_output['cp_ts'], changepoint_output['cp_ts'][1:]):
        cp_ts_temp = [i, j]
        cp_ts = np.append(cp_ts, cp_ts_temp)
    cp_index = []
    for i, j in zip(changepoint_output['cp_index'], changepoint_output['cp_index'][1:]):
        cp_index_temp = [i, j]
        cp_index = np.append(cp_index, cp_index_temp)        
    cp_countrate = []
    for i, j in zip(changepoint_output['cp_countrate'][1:], changepoint_output['cp_countrate'][1:]):
        cp_countrate_temp = [i, j]
        cp_countrate = np.append(cp_countrate, cp_countrate_temp)
    cp_state = []
    for i, j in zip(changepoint_output['cp_state'][1:], changepoint_output['cp_state'][1:]):
        cp_state_temp = [i, j]
        cp_state = np.append(cp_state, cp_state_temp)
    cp_out_cor = pd.DataFrame()
    cp_out_cor['cp_index'] = cp_index
    cp_out_cor['cp_ts'] = cp_ts
    cp_out_cor['cp_state'] = cp_state
    cp_out_cor['cp_countrate'] = cp_countrate
    return cp_out_cor

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
    state_cp = df['cp_state'].values
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
    if len(dig_uniq) == len(countrate_cp):
        df_dig['countrate_cp'] = df_dig['countrate_cp'].replace(dig_uniq, countrate_cp)
    elif len(dig_uniq) != len(countrate_cp):
        df_dig['countrate_cp'] = df_dig['countrate_cp'].replace(dig_uniq[1:], countrate_cp)     
    df_dig['state'] = dig_cp  # initiating array for state assigment
    avg_countrate = np.average(countrate_cp)  # from change point_list
    # avg_countrate = 500
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

def plot_changepoint_trace(ax, timestamps, changepoint_output, bintime,
                           x_lim_min=0, y_lim_min=0, x_lim_max=5, y_lim_max=6,
                           show_changepoint=True):
    ''' help doc
    '''
    # realtime trace plot
    bins = int((max(timestamps) - min(timestamps)) / bintime)
    binned_trace = np.histogram(timestamps, bins=bins)
    ax.plot(binned_trace[1][:-1], 1e-3 *
            binned_trace[0] / bintime, 'b', alpha=0.5)
    # changepoint plot
    cp_out_cor = changepoint_output_corr(changepoint_output)
    if show_changepoint:
        ax.plot(cp_out_cor['cp_ts'], 1e-3*cp_out_cor['cp_countrate'], 'r')
    # axis properties
    ax.set_xlim(x_lim_min, x_lim_max)
    ax.set_ylim(y_lim_min, y_lim_max)
    ax.set_xlabel('Time/s')
    ax.set_ylabel('Counts/kcps')
    return

def onoff_fromCP(cp_out, timestamps):
    def find_closest(A, target):
        # https://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
        #A must be sorted
        idx = A.searchsorted(target)
        idx = np.clip(idx, 1, len(A) - 1)
        left = A[idx - 1]
        right = A[idx]
        idx -= target - left < right - target
        return idx
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
    idx_closest = find_closest(timestamps, time_cp)
    timestamps_closest = timestamps[idx_closest]    
    dig_cp = np.digitize(timestamps, timestamps_closest)
    dig_uniq = np.unique(dig_cp)
    df_dig = pd.DataFrame()
    df_dig['timestamps'] = timestamps    
    df_dig['cp_no'] = dig_cp
    df_dig['countrate_cp'] = dig_cp
    if len(dig_uniq) == len(countrate_cp):
        df_dig['countrate_cp'] = df_dig['countrate_cp'].replace(dig_uniq, countrate_cp)
    elif len(dig_uniq) != len(countrate_cp):
        df_dig['countrate_cp'] = df_dig['countrate_cp'].replace(dig_uniq[1:], countrate_cp)
    # elif len(dig_uniq) < len(countrate_cp):
    #     df_dig['countrate_cp'] = df_dig['countrate_cp'].replace(dig_uniq, countrate_cp[:-1])        
    df_dig['state'] = dig_cp  # initiating array for state assigment
    avg_countrate = np.average(countrate_cp)  # from change point_list
    df_dig['state'][df_dig['countrate_cp'] < avg_countrate] = 1
    df_dig['state'][df_dig['countrate_cp'] > avg_countrate] = 2

    df_on = df_dig[df_dig['state']==2]#.reset_index(drop=True)
    df_off = df_dig[df_dig['state']==1]#.reset_index(drop=True)
    # ontime calc
    time_left = df_on.groupby('cp_no').timestamps.min();
    time_right = df_on.groupby('cp_no').timestamps.max();
    abs_ontime = df_on.groupby('cp_no').timestamps.mean();
    ontimes = time_right - time_left
    # offtime calc
    time_left = df_off.groupby('cp_no').timestamps.min();
    time_right = df_off.groupby('cp_no').timestamps.max();
    abs_offtime = df_off.groupby('cp_no').timestamps.mean();
    offtimes = time_right - time_left
    # average ontime calc
    tonav = np.average(ontimes)
    lambda_ton = 1/tonav;
    lambda_ton_low = lambda_ton * (1-(1.96/np.sqrt(len(ontimes))))
    lambda_ton_upp = lambda_ton * (1+(1.96/np.sqrt(len(ontimes))))
    tonav_err = (1/lambda_ton_low) - (1/lambda_ton_upp);
    tonav_err = np.round(tonav_err, 4)
    # average offtime calc
    toffav = np.average(offtimes);# also converts to millisecond
    lambda_toff = 1/toffav;
    lambda_toff_low = lambda_toff * (1-(1.96/np.sqrt(len(offtimes))))
    lambda_toff_upp = lambda_toff * (1+(1.96/np.sqrt(len(offtimes))))
    toffav_err = (1/lambda_toff_low) - (1/lambda_toff_upp);
    toffav_err = np.round(toffav_err, 4)
    # put it in dataframe
    onoff_out = {'ontimes': ontimes,
                 'abs_ontime': abs_ontime,
                 'offtimes': offtimes,
                 'abs_offtime': abs_offtime,
                 'tonav': tonav,
                 'tonav_err': tonav_err,
                 'toffav': toffav,
                 'toffav_err': toffav_err}
    return onoff_out

def sim_vs_changept(simulatedhdf5, pars=(1, 0.1, 0.9, 2), time_sect=25,
                    range_on=(0, 0.1), bins_on=100,
                    range_off=(0, 0.5), bins_off=100,
                   countrate_max=5):
    h5 = h5py.File(simulatedhdf5, 'r')
    grp_exp = 'exp_changepoint'
    grp_cpars = '/'+ grp_exp + '/cp_'+str(pars[1])+'_'+str(pars[2])+'_'+str(time_sect)+'s'
    cp_out = pd.DataFrame(h5[grp_cpars][:],
                          columns = ['cp_index', 'cp_ts', 'cp_state', 'cp_countrate'])
    timestamps = h5['onexp_offexp']['timestamps'][...]
    t_on_sim = h5['onexp_offexp']['ontimes_exp'][...]
    t_off_sim = h5['onexp_offexp']['offtimes_exp'][...]
    h5.close()
    plt.close('all')
    fig = plt.figure(figsize=(10, 5))
    nrows=2;ncols=2;
    ax00 = plt.subplot2grid((nrows, ncols),(0,0), colspan=2)
    ax10 = plt.subplot2grid((nrows, ncols),(1,0))
    ax11 = plt.subplot2grid((nrows, ncols),(1,1))
    # time trace plot
    plot_changepoint_trace(ax00, timestamps, changepoint_output=cp_out,
                   bintime=5e-3,
                   x_lim_min=0, y_lim_min=0, x_lim_max=10, y_lim_max=countrate_max,
                   show_changepoint=True);
    ax00.set_title(simulatedhdf5 +'\n'+str(pars))
    # on/off from changepoint ananlysis
    out = changepoint_simulatedata(simulatedhdf5, time_sect=time_sect, pars=pars,
                                 exp=True, rise=False, overwrite=False)
    [simulatedhdf5, timestamps, cp_out] = out    
    onoff_out = onoff_fromCP(cp_out, timestamps)
    # on histogram
    t_on_cps = onoff_out['ontimes']#changepoint output
    n, t = np.histogram(t_on_cps, bins=bins_on, range=range_on, density=True)
    ax10.plot(t[:-1], n, label='Bright times\nChange Point')
    n, t = np.histogram(t_on_sim, bins=bins_on, range=range_on, density=True)
    ax10.plot(t[:-1], n, '.', label='Bright times\nSimulated')
    ax10.set_xlim(range_on)
    ax10.set_xlabel('duration/s')
    ax10.set_ylabel('#')
    ax10.legend()
    # off histogram
    t_off_cps = onoff_out['offtimes']
    n, t = np.histogram(t_off_cps, bins=bins_off, range=range_off, density=True)
    ax11.plot(t[:-1], n, label='Dark times\nChange Point')
    n, t = np.histogram(t_off_sim, bins=bins_off, range=range_off, density=True)
    ax11.plot(t[:-1], n, '.', label='Dark times\nSimulated')
    ax11.set_xlim(range_off)
    ax11.set_xlabel('duration/s')
    ax11.set_ylabel('#')
    ax11.legend()
    fig.tight_layout()
    return fig

    # =========== FOLDERWISE ==============
def changepoint_folderwise(folderpath, pars=(1, 0.1, 0.9, 2), overwrite=False):
    start_time = time.time()
    pt3_extension = [".pt3"]
    for dirpath, dirname, filenames in os.walk(folderpath):
        for filename in [f for f in filenames if f.endswith(tuple(pt3_extension))]:
            file_path_pt3 = os.path.join(dirpath, filename)
            file_path_hdf5 = file_path_pt3[:-3] + 'hdf5'
            file_path_datn = file_path_hdf5[:-4] + 'pt3.datn'
            if os.path.isfile(file_path_datn):
                start_time_i = time.time()
                import datetime
                date = datetime.datetime.today().strftime('%Y%m%d_%H%M')
                print("---%s : Changepoint execution started for %s\n" %
                      (date, file_path_hdf5))
                try:
                    df_datn = pd.read_csv(file_path_datn, header=None)
                    tmin = min(df_datn[0])
                    tmax = max(df_datn[0])
                    changepoint_photonhdf5(file_path_hdf5, tmin=tmin,
                                                            tmax=tmax, pars=pars,
                                                            overwrite=overwrite)  # MOST iMPORTANT parameters
                except:
                    #print(file_path_datn)
                    changepoint_photonhdf5(file_path_hdf5, pars=pars,
                                                            overwrite=overwrite)  # MOST iMPORTANT parameters
                processtime = time.time() - start_time_i
                print("---TOTAL time took for the file: %s IS: %s seconds ---\n" %
                      (file_path_hdf5, processtime))
            else:
                print(file_path_datn + ' : doesnot exist\n')
    print("---TOTAL time took for the folder: %s seconds ---\n" %
          (time.time() - start_time))
    return
