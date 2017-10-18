import numpy as np
import matplotlib.pyplot as plt
def simulate_on_off_times(ton1=0.01, ton2=0.002, toff1=0.250, toff2=0.005, time_len=60, plotting=True):
    '''ton, toff1, toff2 are in seconds
    Keep toff2 > toff1
    '''
    #on time pdfs
    k1=1/ton1; k2=1/ton2;
    tons_pdf = np.linspace(0, ton1*10, 1000);
    pdf_on_exp = k1*np.exp(-k1*tons_pdf);
    pdf_on_exp = pdf_on_exp/sum(pdf_on_exp);
    pdf_on_rise = (k1*k2/(k1-k2))*(np.exp(-k2*tons_pdf) - np.exp(-k1*tons_pdf));
    pdf_on_rise = pdf_on_rise/sum(pdf_on_rise);#probability sums to 1
    #off time pdfs
    k3 = 1/toff1; k4=1/toff2;
    toffs_pdf = np.linspace(0, toff1*10, 1000)
    pdf_off_exp = k3*np.exp(-k3*toffs_pdf);
    pdf_off_exp = pdf_off_exp/sum(pdf_off_exp);
    pdf_off_rise = (k3*k4/(k3-k4))*(np.exp(-k4*toffs_pdf) - np.exp(-k3*toffs_pdf));
    pdf_off_rise = pdf_off_rise/sum(pdf_off_rise);#probability sums to 1
    #on times
    numpoints = int(time_len/(ton1+toff1));
    print('numpoint: '+str(numpoints))
    ontimes_exp_1 = np.random.choice(tons_pdf, numpoints, p=pdf_on_exp);
    ontimes_exp_rise = np.random.choice(tons_pdf, numpoints, p=pdf_on_rise);
    offtimes_exp_1 = np.random.choice(toffs_pdf, numpoints, p=pdf_off_exp);
    offtimes_exp_rise = np.random.choice(toffs_pdf, numpoints, p=pdf_off_rise);
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
        time = np.cumsum(ontimes_exp_1+offtimes_exp_1)
        ax00.plot(time, ontimes_exp_1, label='ontimes exponential')
        time = np.cumsum(ontimes_exp_rise+offtimes_exp_rise)
        ax10.plot(time, ontimes_exp_rise, label='ontimes risetime')
        hist_exp, trace_exp = np.histogram(ontimes_exp_1, bins=bins_default)
        hist_rise, trace_rise = np.histogram(ontimes_exp_rise, bins=bins_default)
        ax20.plot(trace_exp[:-1], hist_exp/max(hist_rise), label='ontimes exp: simulated')
        ax20.plot(tons_pdf, pdf_on_exp/max(pdf_on_rise), label='ontimes exp: analytical')
        ax20.plot(trace_rise[:-1], hist_rise/max(hist_rise), label='ontimes rise: simulated')
        ax20.plot(tons_pdf, pdf_on_rise/max(pdf_on_rise), label='ontimes rise: analytical')
        # offtimes exponential
        time = np.cumsum(ontimes_exp_1+offtimes_exp_1)
        ax01.plot(time, offtimes_exp_1, label='offtimes exponential')
        time = np.cumsum(ontimes_exp_rise+offtimes_exp_rise)
        ax11.plot(time, offtimes_exp_rise, label='offtimes risetime')
        hist_exp, trace_exp = np.histogram(offtimes_exp_1, bins=bins_default)
        hist_rise, trace_rise = np.histogram(offtimes_exp_rise, bins=bins_default)
        ax21.plot(trace_exp[:-1], hist_exp/max(hist_rise), label='offtimes exp: simulated')
        ax21.plot(toffs_pdf, pdf_off_exp/max(pdf_off_rise), label='offtimes exp: analytical')
        ax21.plot(trace_rise[:-1], hist_rise/max(hist_rise), label='offtimes rise: simulated')
        ax21.plot(toffs_pdf, pdf_off_rise/max(pdf_off_rise), label='offtimes rise: analytical')
        
        ax00.legend();ax01.legend();
        ax10.legend(); ax11.legend();
        ax20.legend(); ax21.legend();
    return ontimes_exp_1, ontimes_exp_rise, offtimes_exp_1, offtimes_exp_rise
def timetrace_from_onofftrace(ontimes, offtimes, bintime=1e-3,
                             i_on_mu=2000, i_on_sig=1000,
                             i_off_mu=200, i_off_sig=240,
                             plotting=True):
    '''
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
    intensity = np.absolute(intensity)
    intensity = np.round(intensity)#.astype(int);
    return intensity, bintime
