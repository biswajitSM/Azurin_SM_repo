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
                           time_sect=25, pars=[1, 0.1, 0.9, 2],
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
    return file_path_hdf5analysis, timestamps
def changepoint_simulatedata(simulatedhdf5, time_sect=25, pars=[1, 0.1, 0.9, 2],
                             exp=True, rise=False, overwrite=False):
    h5 = h5py.File(simulatedhdf5, 'r+')
    if exp:
        grp_exp = 'exp_changepoint'
        if not '/'+grp_exp in h5.keys():
            h5.create_group(grp_exp)
        timestamps_exp = h5['onexp_offexp']['timestamps'][...]
        grp_cpars = '/'+ grp_exp + '/cp_'+str(pars[1])+'_'+str(pars[2])+'_'+str(time_sect)+'s'
        if grp_cpars in h5.keys() and overwrite:
            print('exist')
            del h5[grp_cpars]
        if not grp_cpars in h5.keys() or overwrite:
            changepoint_output = changepoint_exec(timestamps_exp, simulatedhdf5, 
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
        timestamps_rise = h5['onrise_offrise']['timestamps'][...]
        grp_cpars = '/'+ grp_rise + '/cp_'+str(pars[1])+'_'+str(pars[2])+'_'+str(time_sect)+'s'
        if grp_cpars in h5.keys() and overwrite:
            print('exist')
            del h5[grp_cpars]
        if not grp_cpars in h5.keys() or overwrite:
            changepoint_output = changepoint_exec(timestamps_rise, simulatedhdf5, 
                                              time_sect=time_sect, pars=pars)
            h5[grp_rise][grp_cpars]=changepoint_output
            h5[grp_rise][grp_cpars].attrs['parameters'] = pars
            h5[grp_rise][grp_cpars].attrs['time_sect'] = time_sect
            cp_cols = 'cp_index, cp_ts, cp_state, cp_countrate'
            h5[grp_rise][grp_cpars].attrs['columns'] = cp_cols
        h5.flush()
    h5.close()
    return
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
        changepoint_em2 = pd.read_csv(file_dat_em2, header=None, sep='\s+')
        changepoint_cp = pd.read_csv(file_dat_cp, header=None, sep='\s+')
        cp_index_1 = pd.DataFrame([[0, 0, 0]])
        changepoint_cp = pd.concat([cp_index_1, changepoint_cp], axis=0).reset_index(drop=True)
        changepoint_ts = pd.DataFrame(timestamps_sect[changepoint_cp[0].values] + timestamps_sect_corr)
        changepoint_comb = pd.concat([changepoint_cp[[0]], changepoint_ts[[0]],
                                       changepoint_em2[[0]], changepoint_em2[[1]]], axis=1)
        changepoint_comb.columns = ['cp_index', 'cp_ts', 'cp_state', 'cp_countrate']
        # remove same consecutive states 
        mask = np.diff(changepoint_comb['cp_state'].values)
        mask = np.append(mask, [11])
        mask = mask != 0 # removing repeatative time indices 
        changepoint_comb_cor = changepoint_comb[mask].reset_index(drop=True)
        changepoint_output = pd.concat([changepoint_output, changepoint_comb_cor]).reset_index(drop=True)
    # remove generated files
    os.remove(file_dat_ts)
    os.remove(file_dat_cp)
    os.remove(file_dat_em2)
    os.remove(file_dat_ah)
    os.remove(file_dat_bic)
    os.remove(file_dat_cp0)
    return changepoint_output
def plot_changepoint(changepoint_output, timestamps):
    '''
    '''
    time_cp = pd.concat([changepoint_output['cp_ts'][1:], changepoint_output['cp_ts']], axis=0).reset_index(drop=True)
    countrate_cp = pd.concat([changepoint_output['cp_countrate'][:-1], 
                              changepoint_output['cp_countrate']], axis=0).reset_index(drop=True)
    df = pd.DataFrame({'cp': time_cp.values,
                       'countrate': countrate_cp.values})
    df = df.sort_values(by=['cp'], kind='mergesort').reset_index(drop=True)#'mergesort', 'quicksort', 'heapsort'
    # df.head(15)
    plt.plot(df['cp'][:200], df['countrate'][:200])
    return
# =========== FOLDERWISE ==============
def changepoint_folderwise(folderpath, pars=[1, 0.1, 0.9, 2], overwrite=False):
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