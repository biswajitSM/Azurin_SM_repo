#Import modules
import numpy as np
import pandas as pd
import os.path
from pylab import *
import glob
import os
import re
from xlwt import Workbook
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
global pointnumber


from numpy import sqrt, pi, exp, linspace, loadtxt
from lmfit import  Model, Parameter, Parameters

import matplotlib.pyplot as plt

wb = Workbook()
sheet1 = wb.add_sheet('on_times',cell_overwrite_ok=True)
sheet2 = wb.add_sheet('off_times',cell_overwrite_ok=True)
sheet3 = wb.add_sheet('on_off_times_FCS',cell_overwrite_ok=True)
sheet4 = wb.add_sheet('FCS fit values',cell_overwrite_ok=True)


sheet1.write(0,0,'Potential (mV):')
sheet2.write(0,0,'Potential (mV):')
sheet3.write(0,0,'Potential (mV):')
sheet1.write(1,0,'Point #')
sheet2.write(1,0,'Point #')
sheet3.write(1,0,'Point #')
for i in list(range(1,32)):
    sheet1.write(i+2,0,i)
    sheet2.write(i+2,0,i)
    sheet3.write(i+2,0,i)
    sheet4.write(i+1,0,i)



def T_off_average(f_datn, f_emplot):

    #expt data

    df = pd.read_csv(f_datn, header=None)
    binpts=5000; mi=min(df[0]); ma=mi+10;
    df_hist = histogram(df[0], bins=binpts, range=(mi, ma))

    #change point

    df = pd.read_csv(f_emplot, header=None, sep='\t')
    df_diff= diff(df[0])
    #calculating Ton and Toff
    df_tag = df[[0, 1]];# df_ton = df_ton[1:]
    df_tag = pd.DataFrame([df_tag[0][1:], diff(df_tag[1])]); df_tag = df_tag.T;
    df_tag.columns = [0, 1];
    df_tag = df_tag[df_tag[1] != 0];
    df_tag.reset_index(drop=True, inplace=True);
    if df_tag[1][0] < 0:
        df_tag = df_tag[1:]
        df_tag.reset_index(drop=True, inplace=True);
    df_tag_pos = df_tag[df_tag[1]==max(df_tag[1])];df_tag_pos.reset_index(drop=True, inplace=True);
    df_tag_neg = df_tag[df_tag[1]==min(df_tag[1])];df_tag_neg.reset_index(drop=True, inplace=True);

    df_ton = df_tag_neg[0]-df_tag_pos[0];df_ton.reset_index(drop=True, inplace=True);
    t1=df_tag_pos[0][1:]; t1.reset_index(drop=True, inplace=True);
    t2=df_tag_neg[0]; t1.reset_index(drop=True, inplace=True);
    df_toff = t1 - t2; df_toff = df_toff[:df_toff.shape[0]-2];df_ton.reset_index(drop=True, inplace=True)
    #remove NAN values:
    df_ton = df_ton[~np.isnan(df_ton)];
    df_toff = df_toff[~np.isnan(df_toff)];
    average_ton = np.average(df_ton);
    average_toff = np.average(df_toff);
    return(average_ton, average_toff, df_ton, df_toff)


def FCS_fit(filename,tmin,tmax):
    df_fcs = pd.read_csv(filename, index_col=False, names=None, skiprows=1, header=None, sep='\ ', engine='python');
    df_fcs = df_fcs[df_fcs[0]>=tmin];
    df_fcs = df_fcs[df_fcs[0]<=tmax];
    df_fcs = df_fcs[np.isfinite(df_fcs[2])]
    np.isnan(df_fcs[0]);#removing nan file
    xdata=df_fcs[0];
    ydata=df_fcs[2];
    def biexp(x, A1, t1, A2, t2):
        return((A1*exp(-x/t1))+(A2*exp(-x/t2)))
    def mono_exp(x, A1, t1):
        return(A1*exp(-x/t1))
    try:
        monofit, pcov_mono = curve_fit(mono_exp, xdata, ydata, p0 = [10, 1], bounds=(0.01, np.inf))
    except RuntimeError:
        print('Error - mono exponential curve_fit failed for file: %s' %filename)
        monofit = np.zeros(2)
        
    try:
        bifit, pcov_bi = curve_fit(biexp, xdata, ydata, p0 = [10, 1, 0.5, 1], bounds=(0.01, np.inf))
    except RuntimeError:
        print('Error - bi-exponential curve_fit failed for file: %s' %filename)
        bifit = np.zeros(4)
    if np.count_nonzero(monofit) == 2 and np.count_nonzero(bifit) == 4:
        for i in range(len(xdata)):
            fit_value_mono = mono_exp(xdata,monofit[0],monofit[1])
            fit_value_bi = biexp(xdata,bifit[0],bifit[1],bifit[2],bifit[3])
            av_diff_mono = np.mean(abs(np.subtract(fit_value_mono, ydata)))
            #av. distance from the fit to the real data.
            av_diff_bi = np.mean(abs(np.subtract(fit_value_bi, ydata)))
            if av_diff_mono > av_diff_bi:
                fitvalues = bifit
            else:
                fitvalues = np.append(monofit,[0,0])
    elif np.count_nonzero(monofit) == 2:
        fitvalues = np.append(monofit,[0,0])
    elif np.count_nonzero(bifit) == 2:
        fitvalues = bifit
        
    return(fitvalues)
        
    

def histograms(titel, pot, pointnumbers, specific_potential, rnge_on, rnge_off, bins_on, bins_off, proteins, homedir, max_his_on, max_his_off, x_shift):
    os.chdir(homedir)
    homedir1 = os.getcwd()
    potential, pntnumbers = pot, pointnumbers #creates an array with dimension potential (col) x pntnumbers (rows) 
    on_time = []
    off_time = []
    extensions = [".datn",".dat"]
    all_on_times_array = [[None] * potential for i in range(pntnumbers)]
    all_off_times_array = [[None] * potential for i in range(pntnumbers)]
    df3 = pd.DataFrame(columns=[specific_potential])
    df3_off = pd.DataFrame(columns=[specific_potential])
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(tuple(extensions))]:
            string_1 = 'mV'
            string_2 = 'FCS'
            position_FCS = filename.find(string_2)
            if position_FCS in [-1]: #no FCS in name --> time trace file
                number1 = filename[-11:-10] #first number
                number2 = filename[-10:-9] #second number
                if number1.isdigit(): #check if 1 or 2-digit number
                    pointnumber = int(number1 + number2)
                elif number2.isdigit():
                    pointnumber = int(number2)
                
                position_potential = filename.find(string_1) #determine the place where the potential number is in the filename
                if position_potential in [-1]: #mV does not appear in the name
                    print('Potential in %s is not properly defined.' %filename)
                else:  #determine the potential (between -999 and +999mV)
                    pot_number1 = filename[position_potential-1:position_potential]
                    pot_number2 = filename[position_potential-2:position_potential-1]
                    pot_number3 = filename[position_potential-3:position_potential-2]
                    pot_number4 = filename[position_potential-4:position_potential-3]
                    pot_number5 = filename[position_potential-5:position_potential-4]
                if pot_number2 in ['_']: #filename X_voltagemV_Y
                    potentential = pot_number1
                elif pot_number3 in ['_']:
                    potentential = pot_number2 + pot_number1
                elif pot_number4 in ['_']:
                    potentential = pot_number3 + pot_number2 + pot_number1
                elif pot_number5 in ['_']:
                    potentential = pot_number4 + pot_number3 + pot_number2 + pot_number1
                potentential = int(potentential) #reading the potential
                if potentential == int(specific_potential):
                    if filename.endswith('.datn'):
                        os.chdir(dirpath)
                        f_datn = filename
                        f_emplot = re.sub('.datn$','.datn.em.plot',f_datn)
                        if os.path.isfile(f_emplot):    
                            all_times = T_off_average(f_datn, f_emplot)
                            t_on = all_times[0]
                            t_off = all_times[1]
                            t_on_hist = all_times[2]
                            t_off_hist = all_times[3]
                            sheet1.write(pointnumber+2,1,t_on)
                            sheet2.write(pointnumber+2,1,t_off)
                            on_time = t_on
                            off_time = t_off
                            on_time_df = pd.DataFrame(t_on_hist,columns = [specific_potential])
                            off_time_df = pd.DataFrame(t_off_hist,columns = [specific_potential])
                            df3 = df3.append(on_time_df)
                            df3_off = df3_off.append(off_time_df)
                        else:
                            print("The file %s does not exist" %f_emplot)
                        os.chdir(homedir1)
            
            
    os.chdir('..')       
    sheet1.write(0,1,int(specific_potential))
    sheet2.write(0,1,int(specific_potential)) 
    
    
 
    df_on_shifted = df3.shift(+1) ## shift up
    df_on_shifted.drop(df3.shape[0] - 1,inplace = True)
    df_off_shifted = df3_off.shift(+1) ## shift up
    df_off_shifted.drop(df3_off.shape[0] - 1,inplace = True)
    
    df_on_shifted_x = df3.shift(+x_shift) ## shift up
    df_on_shifted_x.drop(df3.shape[0] - x_shift,inplace = True)
    df_off_shifted_x = df3_off.shift(+x_shift) ## shift up
    df_off_shifted_x.drop(df3_off.shape[0] - x_shift,inplace = True)
  
    
    def single_exp(x_values, constant1,constant2):
        return(constant1*exp(-constant2*x_values))

    def double_exp(x_values, constant3, constant4, constant5, constant6):
        return(constant3*exp(-constant4*x_values) - constant5*exp(-constant6*x_values))
                     
    fig1 = plt.figure(figsize=(12,10))
    plt.xlabel(r'$\tau_{on}$')
    plt.ylabel('#')
    n,bins_on1,patches = hist(df3, range=(0,max_his_on),bins=bins_on)
    bin_centers_on = bins_on1[:-1] + 0.5 * (bins_on1[1:] - bins_on1[:-1])    
    popt_single, pcov_single = curve_fit(single_exp, bin_centers_on, n, p0 = [1, 1])
    plt.plot(bin_centers_on, single_exp(bin_centers_on, *popt_single))
    plt.title('ON time %s-Azu %smV' %(proteins, specific_potential))
    fig1.savefig("ON_histograms_{i}mV.png".format(i=specific_potential))   

    fig10 = plt.figure(figsize=(12,10))
    plt.title('OFF time %s-Azu %smV' %(proteins, specific_potential))
    plt.xlabel(r'$\tau_{off}$')
    plt.ylabel('#')
    n_off,bins_off1,patches_off = hist(df3_off, range=(0,max_his_off),bins=bins_off)
    bin_centers_off = bins_off1[:-1] + 0.5 * (bins_off1[1:] - bins_off1[:-1])    
    popt_double, pcov_double = curve_fit(double_exp, bin_centers_off, n_off, p0 = [1, 1, 1, 1])
    plt.plot(bin_centers_off, double_exp(bin_centers_off, *popt_double))
    fig10.savefig("OFF_histograms_{i}mV.png".format(i=specific_potential))   
    
    print(r'Fit ON time histogram: %s * e^{-%s t}' %(popt_single[0],popt_single[1]))
    print(r'Fit OFF time histogram: %s * e^{-%s t} - %s * e^{-%s t}' %(popt_double[0],popt_double[1],popt_double[2],popt_double[3]))
    

    fig2 = plt.figure(figsize=(12,10))
    ax3 = fig2.add_subplot(2,2,1)
    hist2d(df3[specific_potential],df_on_shifted[specific_potential], range=rnge_on ,bins=bins_on, norm=LogNorm());
    colorbar()
    ax3.set_title('ON time %s-Azu %smV' %(proteins, specific_potential))
    ax3.set_xlabel(r'$\tau_{on}/s$')
    ax3.set_ylabel(r'$\tau_{on}+1/s$')
        
    ax4 = fig2.add_subplot(2,2,2)
    hist2d(df3_off[specific_potential],df_off_shifted[specific_potential], range=rnge_off,bins=bins_off, norm=LogNorm());
    colorbar()
    ax4.set_title('OFF time %s-Azu %smV' %(proteins, specific_potential))
    ax4.set_xlabel(r'$\tau_{off}/s$')
    ax4.set_ylabel(r'$\tau_{off}+1/s$')
    plt.tight_layout()
    
    ax5 = fig2.add_subplot(2,2,3)
    hist2d(df3[specific_potential],df_on_shifted_x[specific_potential], range=rnge_on ,bins=bins_on, norm=LogNorm());
    colorbar()
    ax5.set_title('ON time %s-Azu %smV' %(proteins, specific_potential))
    ax5.set_xlabel(r'$\tau_{on}/s$')
    ax5.set_ylabel(r'$\tau_{off}+%s/s$' %x_shift)
        
    ax6 = fig2.add_subplot(2,2,4)
    hist2d(df3_off[specific_potential],df_off_shifted_x[specific_potential], range=rnge_off,bins=bins_off, norm=LogNorm());
    colorbar()
    ax6.set_title('OFF time %s-Azu %smV' %(proteins, specific_potential))
    ax6.set_xlabel(r'$\tau_{off}/s$')
    ax6.set_ylabel(r'$\tau_{off}+%s/s$' %x_shift)
    plt.tight_layout()
    fig2.savefig("2dhistograms_{i}mV.png".format(i=specific_potential))   

    
    
    fig = plt.figure(figsize=(12,10))
    ax1 = fig.add_subplot(211)
    plt.title('On time') 
    plt.xlabel('potential(mV)')
    plt.ylabel('ms')
    plt.xlim(specific_potential-10,specific_potential+10)
    for i in range(len(df3.index)):
        plt.scatter(list(df3),df3.iloc[i])
    
    ax2 = fig.add_subplot(212)
    plt.title('Off time')
    plt.xlabel('potential(mV)')
    plt.ylabel('ms')
    plt.xlim(specific_potential-10,specific_potential+10)
    for i in range(len(df3_off.index)):
        plt.scatter(list(df3_off),df3_off.iloc[i])
    plt.tight_layout()
    fig.savefig("Cualltimes")
    
    wb.save(titel)

    return()    
    