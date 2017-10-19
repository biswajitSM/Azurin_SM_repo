import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def simulate_on_off_times(ton1=0.01, ton2=0.002, toff1=0.250, toff2=0.005, time_len=60, plotting=False):
    '''ton, toff1, toff2 are in seconds
    Keep toff2 > toff1
    '''
    #on time pdfs
    k1=1/ton1; k2=1/ton2;
    t_pdf = np.random.uniform(0,1, 1e6);
    tons_pdf = t_pdf*ton1*10# np.linspace(0, ton1*10, 100000);
    pdf_on_exp = k1*np.exp(-k1*tons_pdf);
    pdf_on_exp = pdf_on_exp/sum(pdf_on_exp);
    pdf_on_rise = (k1*k2/(k1-k2))*(np.exp(-k2*tons_pdf) - np.exp(-k1*tons_pdf));
    pdf_on_rise = pdf_on_rise/sum(pdf_on_rise);#probability sums to 1
    #off time pdfs
    k3 = 1/toff1; k4=1/toff2;
    toffs_pdf = t_pdf*toff1*10#np.linspace(0, toff1*10, 100000)
    pdf_off_exp = k3*np.exp(-k3*toffs_pdf);
    pdf_off_exp = pdf_off_exp/sum(pdf_off_exp);
    pdf_off_rise = (k3*k4/(k3-k4))*(np.exp(-k4*toffs_pdf) - np.exp(-k3*toffs_pdf));
    pdf_off_rise = pdf_off_rise/sum(pdf_off_rise);#probability sums to 1
    #on times
    numpoints = int(time_len/(ton1+toff1));
    print('numpoint: '+str(numpoints))
    ontimes_exp_1 = np.random.choice(tons_pdf, numpoints, p=pdf_on_exp);
    ontimes_exp_1 = np.round(ontimes_exp_1, 4);
    ontimes_exp_rise = np.random.choice(tons_pdf, numpoints, p=pdf_on_rise);
    ontimes_exp_rise = np.round(ontimes_exp_rise, 4);
    offtimes_exp_1 = np.random.choice(toffs_pdf, numpoints, p=pdf_off_exp);
    offtimes_exp_1 = np.round(offtimes_exp_1, 4);
    offtimes_exp_rise = np.random.choice(toffs_pdf, numpoints, p=pdf_off_rise);
    offtimes_exp_rise = np.round(offtimes_exp_rise, 4);
    if plotting == True:
        bins_default = 100;
        fig = plt.figure(figsize=(20, 10))
        nrows=3; ncols=2;
        ax00 = plt.subplot2grid((nrows, ncols), (0,0)) 
        ax01 = plt.subplot2grid((nrows, ncols), (0,1))
        ax10 = plt.subplot2grid((nrows, ncols), (1,0))
        ax11 = plt.subplot2grid((nrows, ncols), (1,1))
        ax20 = plt.subplot2grid((nrows, ncols), (2,0))
        ax21 = plt.subplot2grid((nrows, ncols), (2,1))
        #ontimes exponential
        time = np.cumsum(ontimes_exp_1 + offtimes_exp_1)
        ax00.plot(time, ontimes_exp_1, label='ontimes exponential')
        time = np.cumsum(ontimes_exp_rise+offtimes_exp_rise)
        ax10.plot(time, ontimes_exp_rise, label='ontimes risetime')
        hist_exp, trace_exp = np.histogram(ontimes_exp_1, bins=bins_default)
        hist_rise, trace_rise = np.histogram(ontimes_exp_rise, bins=bins_default)
        ax20.plot(trace_exp[:-1], hist_exp/max(hist_rise), label='ontimes exp: simulated')
        ax20.plot(tons_pdf, pdf_on_exp/max(pdf_on_rise), '.', ms=0.5, label='ontimes exp: analytical')
        ax20.plot(trace_rise[:-1], hist_rise/max(hist_rise), label='ontimes rise: simulated')
        ax20.plot(tons_pdf, pdf_on_rise/max(pdf_on_rise), '.', ms=0.5, label='ontimes rise: analytical')
        # offtimes exponential
        time = np.cumsum(ontimes_exp_1+offtimes_exp_1)
        ax01.plot(time, offtimes_exp_1, label='offtimes exponential')
        time = np.cumsum(ontimes_exp_rise+offtimes_exp_rise)
        ax11.plot(time, offtimes_exp_rise, label='offtimes risetime')
        hist_exp, trace_exp = np.histogram(offtimes_exp_1, bins=bins_default)
        hist_rise, trace_rise = np.histogram(offtimes_exp_rise, bins=bins_default)
        ax21.plot(trace_exp[:-1], hist_exp/max(hist_rise), label='offtimes exp: simulated')
        ax21.plot(toffs_pdf, pdf_off_exp/max(pdf_off_rise), '.',ms=0.5, label='offtimes exp: analytical')
        ax21.plot(trace_rise[:-1], hist_rise/max(hist_rise), label='offtimes rise: simulated')
        ax21.plot(toffs_pdf, pdf_off_rise/max(pdf_off_rise), '.',ms=0.5, label='offtimes rise: analytical')
        
        ax00.legend();ax01.legend();
        ax10.legend(); ax11.legend();
        ax20.legend(); ax21.legend();
    return ontimes_exp_1, ontimes_exp_rise, offtimes_exp_1, offtimes_exp_rise
def timestamps_from_onofftrace(ontimes, offtimes,
                          i_on_mu=2000, i_off_mu=200,
                          plotting=False):
    '''
    Arguments:
    ontimes, offtimes: array of on and off times in 'seconds' of eqal length
    i_on_mu: average (mean) counts/seconds in on-state (bright)
    i_off_mu: average counts in off state(dim state)
    Return:
    An array of photon arrival times
    If needed, array of interphoton times on on and off times can be generated 
    '''
    t_int = np.random.uniform(0, 1, 1e6);
    #pdf for poissonian on counts
    t_int_on_pdf = 10*t_int/i_on_mu;
    pdf_int_on = i_on_mu * np.exp(-i_on_mu*t_int_on_pdf);
    pdf_int_on = pdf_int_on/sum(pdf_int_on);#Normalization
    #pdf for poissonian on counts
    t_int_off_pdf = 10*t_int/i_off_mu;    
    pdf_int_off = i_off_mu * np.exp(-i_off_mu*t_int_off_pdf);
    pdf_int_off = pdf_int_off/sum(pdf_int_off);
    #number of photons on each on or off levels
    ontimes_counts = np.round(ontimes * i_on_mu);
    offtimes_counts = np.round(offtimes * i_off_mu);
    #geting time stamps
    intphoton_on = []; intphoton_off = [];
    timestamps = []; 
    timestamps_start = 0;
    timestamps_marker=np.array([], dtype=np.uint8);
    for i in range(len(ontimes_counts)):
        on_counts_i = ontimes_counts[i];
        off_counts_i = offtimes_counts[i];
        intphoton_on_i = np.random.choice(t_int_on_pdf, on_counts_i, p=pdf_int_on);
        intphoton_off_i = np.random.choice(t_int_off_pdf, off_counts_i, p=pdf_int_off);
        timestamps_i = np.append(intphoton_on_i, intphoton_off_i);
        timestamps_i = np.cumsum(timestamps_i) + timestamps_start;
        timestamps = np.append(timestamps, timestamps_i);
        timestamps_start = timestamps[-1];
        timestamps_marker_i = np.append(np.ones_like(intphoton_on_i),
                                        np.zeros_like(intphoton_off_i));
        timestamps_marker_i = timestamps_marker_i.astype(np.uint8)
        timestamps_marker = np.append(timestamps_marker, timestamps_marker_i);
        #intphoton_on = np.append(intphoton_on, intphoton_on_i);
        #intphoton_off = np.append(intphoton_off, intphoton_off_i);
    if plotting:
        #plotting the pdfs
        plt.figure()
        plt.plot(t_int_on_pdf, pdf_int_on, '.')
        plt.plot(t_int_off_pdf, pdf_int_off, '.')
    timestamps = np.round(timestamps, 8);
    return timestamps, timestamps_marker
def save_simulated_trace(ton1=0.016, ton2=0.002, toff1=0.250,
                    toff2=0.02, time_len=10, 
                    i_on_mu=3000, i_off_mu=200, 
                    allcomb=False):
    '''
    time traces are simulated with photon arrival times
    all the necessary information are saved in a hdf5 file'''
    #getting name of data folder and creating data folder if does't exist 
    data_folder_sim = 'data_simulated'
    if os.path.isdir(data_folder_sim):
        data_folder_sim = os.path.abspath(data_folder_sim)
    else:
        data_folder_sim = os.makedirs(data_folder_sim)
        data_folder_sim = os.getcwd();
    #creating hdf5 file name where data will be saved
    import datetime
    date = datetime.datetime.today().strftime('%Y%m%d_%H%M')
    hdf5_tosave = os.path.join(data_folder_sim, 
                              date+'_ton'+str(ton1)+'toff'+str(toff1)+'timelen'+str(time_len)
                              +'.hdf5');
    f_saveHDF5 = h5py.File(hdf5_tosave, "w");#create or open hdf5 file in write mode
    #save all the parameters used in the simulation
    f_saveHDF5['parameters/ton1'] = ton1;
    f_saveHDF5['parameters/ton2'] = ton2;
    f_saveHDF5['parameters/toff1'] = toff1;
    f_saveHDF5['parameters/toff2'] = toff2;
    f_saveHDF5['parameters/counts_on'] = i_on_mu;
    f_saveHDF5['parameters/counts_off'] = i_off_mu;
    #Simulate array of on and off times
    out = simulate_on_off_times(ton1, ton2, toff1,
                                toff2, time_len, plotting=False);
    [ontimes_exp_1, ontimes_exp_rise, offtimes_exp_1, offtimes_exp_rise] = out
    # simulate for ontime_exp_rise and offtime_exp_rise;
    grp_rise = f_saveHDF5.create_group('onrise_offrise');
    grp_rise.create_dataset('ontimes_exp_rise', data=ontimes_exp_rise);
    grp_rise.create_dataset('offtimes_exp_rise', data=offtimes_exp_rise);
    out_ts = timestamps_from_onofftrace(ontimes_exp_rise, offtimes_exp_rise, 
                                     i_on_mu, i_off_mu);
    [timestamps, timestamps_marker] = out_ts;
    grp_rise.create_dataset('timestamps', data=timestamps);
    grp_rise.create_dataset('timestamps_marker', data=timestamps_marker, dtype='int8');
    f_saveHDF5.flush()
    # simulate for ontime_exp1 and offtime_exp_1
    grp_exp = f_saveHDF5.create_group('onexp_offexp');
    grp_exp.create_dataset('ontimes_exp', data=ontimes_exp_1);
    grp_exp.create_dataset('offtimes_exp', data=offtimes_exp_1);
    out_ts = timestamps_from_onofftrace(ontimes_exp_1, offtimes_exp_1, 
                                     i_on_mu, i_off_mu);
    [timestamps, timestamps_marker] = out_ts;
    grp_exp.create_dataset('timestamps', data=timestamps);
    grp_exp.create_dataset('timestamps_marker', data=timestamps_marker, dtype='int8');
    f_saveHDF5.flush()
    # close hdf5 and end of func
    f_saveHDF5.close()
    return hdf5_tosave
def timetrace_from_onofftrace(ontimes, offtimes, bintime=1e-3,
                             i_on_mu=2000, i_on_sig=1000,
                             i_off_mu=200, i_off_sig=240,
                             plotting=True):
    '''
    !!! Works well high countrate trace e.g ~1e4 kcps
    Arguments:
    ontimes and offtimes should be given in 'seconds' (unit) and they should be equal length
    bintime: in sec
    i_on_mu, i_off_mu: intensity for on and off levels in counts/sec
    i_on_sig, i_off_sig: intensity-variance for on and off levels in counts/sec
    '''
    #counts per bintime
    counts_on_mu = bintime * i_on_mu;
    counts_on_sig = bintime * i_on_sig;
    counts_off_mu = bintime * i_off_mu;
    counts_off_sig = bintime * i_off_sig;
    # times to milliseconds and converting to integer for generating binned trace
    ontimes = np.round(ontimes/bintime);
    offtimes = np.round(offtimes/bintime);
    intensity = [];
    ontimes_upd=[]; offtimes_upd=[];
    for i in range(len(ontimes)):
        t_on = ontimes[i];
        t_off = offtimes[i];
        if t_on!=0 and t_off!=0:
            #print('remove the on and off time')
            on_int_temp = np.random.normal(counts_on_mu, counts_on_sig, t_on);#generate intensity
            off_int_temp = np.random.normal(counts_off_mu, counts_off_sig, t_off)
            intensity = np.append(intensity, off_int_temp);
            intensity = np.append(intensity, on_int_temp);
            ontimes_upd = np.append(ontimes_upd, t_on);
            offtimes_upd = np.append(offtimes_upd, t_off);
    intensity = np.absolute(intensity)
    intensity = np.round(intensity)#.astype(int);
    if plotting:
        time_bin = bintime * np.linspace(0, len(intensity), len(intensity))
        fig = plt.figure(figsize=(10, 5))
        nrows=2; ncols=2;
        ax00 = plt.subplot2grid((nrows, ncols), (0,0))
        ax01 = plt.subplot2grid((nrows, ncols), (0,1))
        ax10 = plt.subplot2grid((nrows, ncols), (1,0))
        ax11 = plt.subplot2grid((nrows, ncols), (1,1))

        ax00.plot(time_bin, intensity*1e-3/bintime);
        # mask = intensity != 0
        ax01.hist(intensity, bins=100);
        ax01.set_yscale('log');
        hist,trace = np.histogram(ontimes, bins=20);
        ax10.plot(trace[:-1], hist, label='ontime hist')
        hist,trace = np.histogram(ontimes_upd*bintime, bins=20)
        ax10.plot(trace[:-1], hist, '.', label='updated ontime hist')

        hist,trace = np.histogram(offtimes, bins=200)
        ax11.plot(trace[:-1], hist, label='offtime hist')
        hist,trace = np.histogram(offtimes_upd*bintime, bins=200)
        ax11.plot(trace[:-1], hist, '.', label='updated offtime hist');

        ax10.legend();ax11.legend()        
    return intensity, bintime, ontimes_upd, offtimes_upd
def intensity_fit(phtoton_arrtimes, bintime):
    '''
    '''
    df = phtoton_arrtimes;
    del phtoton_arrtimes;
    tt_length=max(df)-min(df);
    tt_length = round(tt_length, 0);
    binpts=tt_length/bintime;
    hist, trace = np.histogram(df, bins=binpts, range=(min(df), max(df)));
    count_hist, counts = np.histogram(hist*1e-3/bintime, bins=100);
    #=======plotting==========
    fig = plt.figure(figsize=(16, 8));
    nrows=2; ncols=2;
    ax00 = plt.subplot2grid((nrows, ncols), (0,0));
    ax01 = plt.subplot2grid((nrows, ncols), (0,1));
    ax10 = plt.subplot2grid((nrows, ncols), (1,0), colspan=2)

    ax00.plot(trace[:-1], hist*1e-3/bintime);
    ax00.set_ylabel('counts/kcps')
    ax00.set_xlabel('time/s')
    ax00.set_title('bintime: ' + str(bintime))
    ax01.bar(counts[:-1], count_hist, width=0.1)
    # mu1, sig1, mu2, sig2 = two_gaussian_fit(ax01, counts[:-1], count_hist)
    ax01.set_yscale('log')
    ax01.set_xlabel('counts/kcps')    
    # ax01.set_title('mu1: '+str(np.round(mu1, 1)) +
    #               '    sig1: '+str(np.round(sig1, 1)) + '\n'
    #                 'mu2: '+str(np.round(mu2, 1)) +
    #               '    sig2: '+str(np.round(sig2, 1)))
    ax10.plot(trace[:-1], hist);
    ax10.set_xlim(0, 10)
    return    
def two_gaussian_fun(x, a1, b1, c1, a2, b2, c2):
    '''
    b = mean; sigma=c;
    '''
    gauss1 = a1*np.exp((-(x-b1)**2)/(2*c1**2))
    gauss2 = a2*np.exp((-(x-b2)**2)/(2*c2**2))
    return gauss1 + gauss2
def two_gaussian_fit(axis, x, pdf):
    fit, pcov = curve_fit(two_gaussian_fun, xdata=x, ydata=pdf,
                          p0=[10, 2, 5, 11, 30, 20], bounds=(-np.inf, np.inf));
    mu1 = fit[1]; sig1 = fit[2];
    mu2 = fit[4]; sig2 = fit[5]
    fitrange = np.linspace(min(x), max(x), 100);
    #fitrange = x;
    axis.bar(x, pdf);
    axis.plot(fitrange, two_gaussian_fun(fitrange, *fit))
    return mu1, sig1, mu2, sig2