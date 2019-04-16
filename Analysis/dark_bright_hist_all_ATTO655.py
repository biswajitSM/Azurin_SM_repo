import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Analysis import *
from ChangePointProcess import *
from pylab import *
matplotlib.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.size"] = "12"
rc('axes', linewidth=1)

#directories
import os
try:
    parentdir
except NameError:
    parentdir = os.getcwd()
else:
    parentdir = parentdir
# homedir=r'/home/biswajit/Research/Reports_ppt/reports/AzurinSM-MS4/data/201702_S101toS104/S101d14Feb17_60.5_635_A2_CuAzu655';#data directory
print('The working directory is parentdir: %s' % parentdir)
# print('The data directory is homedir: %s' %homedir)

import warnings
warnings.filterwarnings("ignore")


#list of folders and their directories CuAZUATTO655
Analysis_dir = '/home/biswajit/Research/Reports_ppt/reports/AzurinSM-MS4/Azurin_SM_repo/Analysis'
# GIVE the PATH of this FOLDER.
data_dir = '/home/biswajit/Research/Reports_ppt/reports/AzurinSM-MS4/data'

# rest automatically created
allfolders = [os.path.join(data_dir, 'AzurinATTO655/CuAzuATTO655/201702_S101toS104/S101d14Feb17'),
              os.path.join(data_dir, 'AzurinATTO655/CuAzuATTO655/201702_S101toS104/S101d15Feb17'),
              os.path.join(data_dir, 'AzurinATTO655/CuAzuATTO655/201702_S101toS104/S101d15Feb17_62.2_635_A2_CuAzu655_2nd'),
              os.path.join(data_dir, 'AzurinATTO655/CuAzuATTO655/201702_S101toS104/S101d16Feb17'),
              os.path.join(data_dir, 'AzurinATTO655/CuAzuATTO655/201702_S101toS104/S104d20Feb17'),
              os.path.join(data_dir, 'AzurinATTO655/CuAzuATTO655/201702_S101toS104/S104d21Feb17'),
              os.path.join(data_dir, 'AzurinATTO655/CuAzuATTO655/201702_S101toS104/S104d21Feb17_60.5_635_A2_CuAzu655'),
              os.path.join(data_dir, 'AzurinATTO655/CuAzuATTO655/20160907_CuAzu655/S81d7Sept16_A2'),
              os.path.join(data_dir, 'AzurinATTO655/CuAzuATTO655/20160907_CuAzu655/S81d7Sept16_A3'),
              os.path.join(data_dir, 'AzurinATTO655/CuAzuATTO655/20160907_CuAzu655/S81d7Sept16_A5'),
              os.path.join(data_dir, 'AzurinATTO655/CuAzuATTO655/20160910_CuAzu655/S83d10Sept16_A3'),
              os.path.join(data_dir, 'AzurinATTO655/CuAzuATTO655/S105d15May17_longtime'),
              os.path.join(data_dir, 'AzurinATTO655/CuAzuATTO655/S106d18May17_longtime'),
              ]
# Checking or creating temp_dir
temp_dir = 'temp'
temp_dir = os.path.join(parentdir, temp_dir)
if os.path.isdir(temp_dir):
    print('temp_dir exists')
else:
    os.makedirs(temp_dir)
    print('temp_dir created')

# functions
def substrate_conc(E0, E, M, n=1):
    '''All potential values are in mV'''
    #E = E + 240; E0=E0+240;
    Ecorr = 10**((E-E0)*n/59) # DUBIOUS formula
    S_oxd = (M*Ecorr)/(1+Ecorr)
    S_red = M/(1+Ecorr)
    return E, S_oxd, S_red #E-240
E=np.arange(0, 200)
M_FeCN = 200;
E, FeCN_oxd, FeCN_red = substrate_conc(E0=180, E=E, M=M_FeCN)
M_asc = 100;
E, asc_oxd, asc_red = substrate_conc(E0=30, E=E, M=M_asc, n=1)
#----------------wrapping vlaues to dataframe------------
def wrap_values():
    df = pd.DataFrame()
    df['E'] = E;
    df['FeCN_oxd'] = FeCN_oxd;
    df['asc_oxd'] = asc_oxd;
    df['FeCN_oxd+asc_oxd'] = FeCN_oxd + asc_oxd;
    df['FeCN_red'] = FeCN_red;
    df['asc_red'] = asc_red;
    df['FeCN_red+asc_red'] = FeCN_red+asc_red;
    return(df)
df=wrap_values()
#--------------Plotting simulated values---------------
def plot_substrate_conc():
    fig, axes = plt.subplots(figsize=(5, 5))
    # FeCN oxd
    ax1 = axes
    ax1.plot(E, FeCN_oxd, 'b', label='$[Fe^{3+}]$')
    ax1.set_ylabel('$[uM]$', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('$Potential[mV]$')

    # reductant
    ax2 = axes
    ax2.plot(E, FeCN_red, 'r', label='$[Fe^{2+}]$')
    ax2.set_xlabel('$Potential[mV]$')

    ax1.legend(loc=4)
    fig.tight_layout()
    plt.show()
    return(fig)
# fig=plot_substrate_conc()
# os.chdir(parentdir)
# os.chdir(temp_dir)
# fig.savefig('substrate_conc.svg', dpi=300, transparent=True)

#Bright dark values
def BrightDarkValuesAll(potential_list=np.array([50, 60, 75, 80, 90, 100]),
                        folderlist=allfolders,
                        pars = (1, 0.01, 0.99, 2),
                        FileToSave = 'S130d02MAr18_all_0.01_0.99.xlsx',
                        Rewrite=False):
    if not os.path.isfile(FileToSave) or Rewrite:
        pot_update = np.array([])
        tonav_update=np.array([])
        toffav_update = np.array([])
        tonstd_update = np.array([])
        toffstd_update = np.array([])
        df_bright = dict()
        df_dark = dict()
        for i in range(len(potential_list)):
            potential=[potential_list[i]]
            t_ons, t_offs = on_off_all_folder(folderlist= folderlist, input_potential=potential, 
                                              pointnumbers=range(100), pars=pars)
            t_ons = t_ons[t_ons < 10]
            t_ons = t_ons[t_ons > 0]
            t_offs = t_offs[t_offs < 10]
            t_offs = t_offs[t_offs > 0]
            df_bright[str(potential[0])+'mV'] = t_ons
            df_dark[str(potential[0])+'mV'] = t_offs

            pot_update = np.append(pot_update, potential[0])
            # average ontime calc
            tonav = np.average(t_ons)
            lambda_ton = 1/tonav;
            lambda_ton_low = lambda_ton * (1-(1.96/np.sqrt(len(t_ons))))
            lambda_ton_upp = lambda_ton * (1+(1.96/np.sqrt(len(t_ons))))
            tonav_err = (1/lambda_ton_low) - (1/lambda_ton_upp);
            tonav_err = np.round(tonav_err, 4)
            # average offtime calc
            toffav = np.average(t_offs);# also converts to millisecond
            lambda_toff = 1/toffav;
            lambda_toff_low = lambda_toff * (1-(1.96/np.sqrt(len(t_offs))))
            lambda_toff_upp = lambda_toff * (1+(1.96/np.sqrt(len(t_offs))))
            toffav_err = (1/lambda_toff_low) - (1/lambda_toff_upp);
            toffav_err = np.round(toffav_err, 4)

            tonav_update = np.append(tonav_update, tonav)
            toffav_update = np.append(toffav_update, toffav)

            tonstd_update = np.append(tonstd_update, tonav_err)
            toffstd_update = np.append(toffstd_update, toffav_err)
        df_avg = pd.DataFrame()
        df_avg['Potential'] = pot_update
        df_avg['average bright time'] = tonav_update
        df_avg['std bright time'] = tonstd_update
        df_avg['avergae dark time'] = toffav_update
        df_avg['std dark time'] = toffstd_update
        #
        df_dark = pd.DataFrame.from_dict(df_dark, orient='index')
        df_dark = df_dark.transpose()
        df_bright = pd.DataFrame.from_dict(df_bright, orient='index')
        df_bright = df_bright.transpose()
        
        E, FeCN_oxd, FeCN_red = substrate_conc(E0=180, E=potential_list, M=200e-6)
        df_avg['FeCN_oxd'] = FeCN_oxd
        df_avg['FeCN_red'] = FeCN_red
        # save to file
        writer = pd.ExcelWriter(FileToSave)
        df_avg.to_excel(writer,'Average', index=False)
        df_bright.to_excel(writer, 'BrightTimes', index=False)
        df_dark.to_excel(writer, 'DarkTimes', index=False)
        writer.save()        
    return FileToSave


# Extract Bright and dark times from the file or create
os.chdir(parentdir)
# file_save = BrightDarkValuesAll(folderlist=allfolders, pars = (1, 0.01, 0.99, 2), FileToSave = 'ATTO655_all_0.01_0.99.xlsx', Rewrite=False)
file_save = BrightDarkValuesAll(folderlist=allfolders, pars = (1, 0.1, 0.9, 2), FileToSave = 'ATTO655_all_0.1_0.9.xlsx', Rewrite=False)

df_avg = pd.read_excel(file_save, sheet_name='Average')
potential_list = df_avg['Potential'].values
tonav_update = df_avg['average bright time'].values
tonstd_update = df_avg['std bright time'].values
# toffav_update = df_avg['avergae dark time'].values
# toffstd_update = df_avg['std dark time'].values
FeCN_oxd = df_avg['FeCN_oxd'].values
FeCN_red = df_avg['FeCN_red'].values
# # bright and dark times histogram
df_bright = pd.read_excel(file_save, sheet_name='BrightTimes')
df_dark = pd.read_excel(file_save, sheet_name='DarkTimes')
ontimes = df_bright['100mV'].dropna().values
# offtimes = df_dark['100mV'].dropna().values
# df_avg

# file_save = BrightDarkValuesAll(folderlist=S130d02MAr18_all, pars = (1, 0.1, 0.9, 2), FileToSave = 'S130d02MAr18_all_0.1_0.9.xlsx', Rewrite=False)

df_avg = pd.read_excel(file_save, sheet_name='Average')
potential_list = df_avg['Potential'].values
# tonav_update = df_avg['average bright time'].values
# tonstd_update = df_avg['std bright time'].values
toffav_update = df_avg['avergae dark time'].values
toffstd_update = df_avg['std dark time'].values
FeCN_oxd = df_avg['FeCN_oxd'].values
FeCN_red = df_avg['FeCN_red'].values
# # bright and dark times histogram
df_bright = pd.read_excel(file_save, sheet_name='BrightTimes')
df_dark = pd.read_excel(file_save, sheet_name='DarkTimes')
# ontimes = df_bright['100mV'].dropna().values
offtimes = df_dark['100mV'].dropna().values
# df_avg


def waitime_hist_inset(waitimes, axis, bins, binrange,
                       insetrange, fit_func, PlotInset):
    '''waitimes: list of on-times or off-times
    '''
    n, bins_hist = np.histogram(waitimes, bins=bins,
                                range=(binrange[0], binrange[1]))  # avoiding zero
    t = bins_hist[:-1]
    n = n[:]
    t_fit = np.linspace(binrange[0], binrange[1], 1000)
    binwidth = np.mean(np.diff(t))
    #fit
    if fit_func.__code__.co_code == mono_exp.__code__.co_code:
        p0 = [10, 1.1]
    elif fit_func.__code__.co_code == risetime_fit.__code__.co_code:
        p0 = [10, 1.1, 0.1]
    elif fit_func.__code__.co_code == streched_exp.__code__.co_code:
        p0 = [10, 0.5, 1000]
    fit, pcov = curve_fit(fit_func, t, n, p0=p0, bounds=(0, np.inf))
    print('k1:' + str(fit[0]))
    print(fit)
    #plot as bar
    from matplotlib import pyplot, transforms
    rot = transforms.Affine2D().rotate_deg(90)
    axis.bar(t, n, width=binwidth, color='k', alpha=0.5)
    axis.plot(t_fit, fit_func(t_fit, *fit), 'k',
              lw=1, label='k1:' + str(fit[0]) + '\n' + str(fit[1]))
    axis.set_xlim(0, None)
    axis.set_ylim(1e0, None)
    axis.set_yscale('log')
    #inset
    if PlotInset:
        axis_in = inset_axes(axis, height="50%", width="50%")
        axis_in.bar(t, n, width=binwidth, color='k', alpha=0.5)
        axis_in.plot(t, n, drawstyle='steps-mid', lw=0.5, color='k')
        axis_in.plot(t_fit, fit_func(t_fit, *fit), 'k',
                     lw=1, label='k1:' + str(fit[0]) + '\n' + str(fit[1]))
        axis_in.set_xlim(insetrange)
        axis_in.get_yaxis().set_ticklabels([])


def risetime_fit(t, k1, k2, A):
    return ((A * k1 * k2 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t)))


def mono_exp(t, k1, A):
    return A * np.exp(-k1 * t)


def streched_exp(t, k, b, A):
    return A * np.exp(-(k * t)**b)


def gaussian(x, amp, cen, sig):
    return amp * np.exp(-(x - cen)**2 / sig**2)


# from LongTraceAnalysis import waitime_hist_inset, streched_exp
# bright and dark times histogram
potential = 100
df_bright = pd.read_excel(file_save, sheet_name='BrightTimes')
ontimes = df_bright[str(potential) + 'mV'].dropna().values
offtimes = df_dark[str(potential) + 'mV'].dropna().values

plt.close('all')
fig = plt.figure(figsize=(10, 3))
nrows = 1
ncols = 2
ax00 = plt.subplot2grid((nrows, ncols), (0, 0))
ax01 = plt.subplot2grid((nrows, ncols), (0, 1))
waitime_hist_inset(ontimes[1:], ax00, bins=50, binrange=[0.01, 0.15], insetrange=[0, 0.01],
                   fit_func=streched_exp, PlotInset=False)
waitime_hist_inset(offtimes, ax01, bins=50, binrange=[0.05, 10], insetrange=[0, 0.1],
                   fit_func=streched_exp, PlotInset=False)
ax00.set_xlabel('bright time/s')
ax00.set_ylabel('#')
ax01.set_xlabel('dark time/s')
ax01.set_ylabel('#')
plt.savefig(os.path.join(temp_dir, 'bright_dark_hist_100mV.svg'),
            format='svg', dpi=300, transparent=True)
print('The figure saved in :' +
      os.path.join(temp_dir, 'bright_dark_hist_100mV.svg'))
plt.show()
# bright time
# dark time
beta = 0.53
k = 694
# (tau_0/beta)* gamma(1/beta)
tau = ((1 / k) / beta) * scipy.special.gamma(beta)
print(tau)
# dark time
beta = 0.75
k = 3
# (tau_0/beta)* gamma(1/beta)
tau = ((1 / k) / beta) * scipy.special.gamma(beta)
print(tau)
