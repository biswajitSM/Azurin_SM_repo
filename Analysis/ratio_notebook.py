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
from matplotlib import gridspec
global pointnumber


from numpy import sqrt, pi, exp, linspace, loadtxt
from lmfit import  Model, Parameter, Parameters

import matplotlib.pyplot as plt




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

def FCS_bi_fit(filename,tmin,tmax):
    df_fcs = pd.read_csv(filename, index_col=False, names=None, skiprows=1, header=None, sep='\ ', engine='python');
    df_fcs = df_fcs[df_fcs[0]>=tmin];
    df_fcs = df_fcs[df_fcs[0]<=tmax];
    df_fcs = df_fcs[np.isfinite(df_fcs[2])]
    np.isnan(df_fcs[0]);#removing nan file
    xdata=df_fcs[0];
    ydata=df_fcs[2];
    def biexp(x, A1, t1, A2, t2):
        return((A1*exp(-x/t1))+(A2*exp(-x/t2)))
    
    bifit, pcov_bi = curve_fit(biexp, xdata, ydata, p0 = [10, 1, 0.5, 1], bounds=(0.01, np.inf))
    return(bifit)

def FCS_mono_fit(filename,tmin,tmax):
    df_fcs = pd.read_csv(filename, index_col=False, names=None, skiprows=1, header=None, sep='\ ', engine='python');
    df_fcs = df_fcs[df_fcs[0]>=tmin];
    df_fcs = df_fcs[df_fcs[0]<=tmax];
    df_fcs = df_fcs[np.isfinite(df_fcs[2])]
    np.isnan(df_fcs[0]);#removing nan file
    xdata=df_fcs[0];
    ydata=df_fcs[2];
    def mono_exp(x, A1, t1):
        return((A1*exp(-x/t1)))
    
    monofit, pcov_mono = curve_fit(mono_exp, xdata, ydata, p0 = [10, 1], bounds=(0.01, np.inf))
    return(monofit)



def t_ratio_notebook(pot, pointnumbers, homedir, x_prots, prot_number_input, save_filename, sav_filename, imp_pot, tminFCS, tmaxFCS, minimal_points):
    wb = Workbook(save_filename)
    sheet1 = wb.add_sheet('t_ratio',cell_overwrite_ok=True)
    sheet1.write(0,0,'Potential (mV):')
    sheet2 = wb.add_sheet('t_ratio_FCS',cell_overwrite_ok=True)
    sheet2.write(0,0,'Potential (mV):')

    os.chdir(homedir)
    homedir1 = os.getcwd()
    potential, pntnumbers = pot, pointnumbers #creates an array with dimension potential (col) x pntnumbers (rows) 
    on_time = [[None] * potential for i in range(pntnumbers)]
    off_time = [[None] * potential for i in range(pntnumbers)]
    t_ratio = [[None] * potential for i in range(pntnumbers)]
    t_ratio_FCS = [[None] * potential for i in range(pntnumbers)]
    extensions = [".datn",".dat"]
    potential_array = []
    potential_array_FCS = []

    all_on_times_array = [[None] * potential for i in range(pntnumbers)]
    all_off_times_array = [[None] * potential for i in range(pntnumbers)]
    df3 = pd.DataFrame(columns=[potential_array])
    df3_off = pd.DataFrame(columns=[potential_array])
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
                if potentential not in potential_array:
                    potential_array.append(potentential) #array with unique potentials
                    all_on_times_array.append(potentential)
                    df3[potentential] = np.nan
                    df3_off[potentential] = np.nan
                for i in range(len(potential_array)): #put the on/off times in the right column with the corresponding potential
                    if potentential == potential_array[i]:
                        if potentential >= imp_pot:
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
                                    t_ratio_calc = t_off / t_on
                                    sheet1.write(pointnumber+2,i+1,t_ratio_calc)

                                    #print(t_ratio_calc)
                                    on_time[pointnumber-1][i] = t_on
                                    off_time[pointnumber-1][i] = t_off
                                    t_ratio[pointnumber-1][i] = t_ratio_calc
                                    on_time_df = pd.DataFrame(t_on_hist,columns = [potential_array[i]])
                                    off_time_df = pd.DataFrame(t_off_hist,columns = [potential_array[i]])
                                                                 
                                    df3 = df3.append(on_time_df)
                                    df3_off = df3_off.append(off_time_df)
                                else:
                                    print("The file %s does not exist" %f_emplot)
                                os.chdir(homedir1)
            else:
                point_number1_FCS = filename[position_FCS-2:position_FCS-1]
                point_number2_FCS = filename[position_FCS-3:position_FCS-2]
                if point_number2_FCS in ['_']: #filename X_voltagemV_Y
                    pointnumberFCS = int(point_number1_FCS)
                else:
                    pointnumberFCS = int(point_number2_FCS + point_number1_FCS)
                position_potential_FCS = filename.find(string_1)
                if position_potential_FCS in [-1]: #mV does not appear in the name
                    print('Potential in %s is not properly defined.' %filename)
                else:  #determine the potential (between -999 and +999mV)
                    pot_number1_FCS = filename[position_potential_FCS-1:position_potential_FCS]
                    pot_number2_FCS = filename[position_potential_FCS-2:position_potential_FCS-1]
                    pot_number3_FCS = filename[position_potential_FCS-3:position_potential_FCS-2]
                    pot_number4_FCS = filename[position_potential_FCS-4:position_potential_FCS-3]
                    pot_number5_FCS = filename[position_potential_FCS-5:position_potential_FCS-4]
                if pot_number2_FCS in ['_']: #filename X_voltagemV_Y
                    potentential_FCS = pot_number1_FCS
                elif pot_number3_FCS in ['_']:
                    potentential_FCS = pot_number2_FCS + pot_number1_FCS
                elif pot_number4_FCS in ['_']:
                    potentential_FCS = pot_number3_FCS + pot_number2_FCS + pot_number1_FCS
                elif pot_number5_FCS in ['_']:
                    potentential_FCS = pot_number4_FCS + pot_number3_FCS + pot_number2_FCS + pot_number1_FCS
                potential_FCS = int(potentential_FCS) #reading the potential
                if potential_FCS not in potential_array:
                    potential_array.append(potential_FCS) #array with unique potentials
                for j in range(len(potential_array)):
                    
                    if potential_FCS == potential_array[j]:
                        if potential_FCS < imp_pot:
                            os.chdir(dirpath)
                            variables_fit = FCS_bi_fit(filename,tminFCS,tmaxFCS)
                            var1 = variables_fit[0] #A1
                            var2 = variables_fit[1] #t1
                            var3 = variables_fit[2] #A2
                            var4 = variables_fit[3] #t2
                            ton_1 = var2*(1+(1/var1))
                            toff_1 = var2*(1+var1)
                            ton_2 = var4*(1+(1/var3))
                            toff_2 = var4*(1+var3)
                            ratio_1 = toff_1 / ton_1
                            ratio_2 = toff_2 /ton_2
                            if ton_1 > ton_2:
                                t_ratio[pointnumberFCS-1][j] = ratio_1
                                sheet1.write(pointnumberFCS+2,j+1,ratio_1)
                                t_ratio_FCS[pointnumberFCS-1][j] = ratio_1
                                sheet2.write(pointnumberFCS+2,j+1,ratio_1)

                            else:
                                t_ratio[pointnumberFCS-1][j] = ratio_2
                                sheet1.write(pointnumberFCS+2,j+1,ratio_2)
                                t_ratio_FCS[pointnumberFCS-1][j] = ratio_2
                                sheet2.write(pointnumberFCS+2,j+1,ratio_2)

                        else:
                            os.chdir(dirpath)
                            variables_fit = FCS_mono_fit(filename,tminFCS,tmaxFCS)
                            var1 = variables_fit[0] #A1
                            var2 = variables_fit[1] #t1
                            ton_1 = var2*(1+(1/var1))
                            toff_1 = var2*(1+var1)
                            ratio_1 = toff_1 / ton_1
                            t_ratio_FCS[pointnumberFCS-1][j] = ratio_1



                            
                            

                                
    
                  
                        
                        os.chdir(homedir1)
    os.chdir(homedir1)
    os.chdir('..')

    for i in list(range(1,pointnumbers+1)):
        sheet1.write(i+2,0,i)
        sheet2.write(i+2,0,i)

    for i in range(len(potential_array)):
        sheet1.write(0,1+i,int(potential_array[i]))
        sheet2.write(0,1+i,int(potential_array[i]))

    
    df_t_ratio = pd.DataFrame(data = t_ratio, index = range(1,pntnumbers+1), columns = potential_array)
    df_t_ratio_FCS = pd.DataFrame(data = t_ratio_FCS, index = range(1,pntnumbers+1), columns = potential_array)

    

    
    
    #df_t_ratio.drop(50, axis=1, inplace=True) #only for the day where 50 mV was not good data
    #df_t_ratio_FCS.drop(50, axis=1, inplace=True) #only for the day where 50 mV was not good data
    #potential_array.remove(50)
    average_ratio_1 =[]
    average_ratio_FCS = []
    for i in potential_array:
        average_ratio_1.append(df_t_ratio[i].mean())
        average_ratio_FCS.append(df_t_ratio_FCS[i].mean())
    
    potential_array[:] = [x / 1000 for x in potential_array]



    
    def nernst(x, a):
        return(10**((x - a) / 0.059))
    
    midpoint_potential_array = []
    for j in range(pointnumbers):
        list_1 = []
        potential_1 = []
        for i in range(len(potential_array)):
            if t_ratio[j][i] is not None:
                list_1.append(t_ratio[j][i])
                potential_1.append(potential_array[i])                
        if len(list_1) >= minimal_points:
            fit_waardes, fit_variance = curve_fit(nernst, potential_1, list_1, p0 = 0.020)
            midpoint_potential_array.append(fit_waardes[0])
            
        del list_1[:]
        del potential_1[:]
    
    av_pot_timetrace = sum(midpoint_potential_array)/len(midpoint_potential_array)
    print(midpoint_potential_array)
    midpoint_potential_array_FCS = []
    for j in range(pointnumbers):
        list_1_FCS = []
        potential_1_FCS = []
        for i in range(len(potential_array)):
            if t_ratio_FCS[j][i] is not None:
                list_1_FCS.append(t_ratio_FCS[j][i])
                potential_1_FCS.append(potential_array[i])                
        if len(list_1_FCS) >= minimal_points:
            fit_waardes_FCS, fit_variance_FCS = curve_fit(nernst, potential_1_FCS, list_1_FCS, p0 = 0.020)
            midpoint_potential_array_FCS.append(fit_waardes_FCS[0])
            
        del list_1[:]
        del potential_1[:]
    
    print(midpoint_potential_array_FCS)
    av_pot_FCS = sum(midpoint_potential_array_FCS)/len(midpoint_potential_array_FCS)   


    '''
    
    popt_0, pcov_0 = curve_fit(nernst, potential_array, average_ratio_1, p0 = 0.020, bounds = (0,np.inf))
    popt_FCS, pcov_FCS = curve_fit(nernst, potential_array, average_ratio_FCS, p0 = 0.020, bounds = (0,np.inf))
    '''
    fig = plt.figure(figsize=(12,10))
    plt.title(r'$ \tau_{off} \tau_{on}^{-1}$ vs potential') 
    plt.xlabel('potential(mV)')
    plt.yscale('log')
    plt.ylabel(r'$\tau_{off}\tau_{on}^{-1}$')
    
    plt.plot(potential_array, nernst(potential_array, av_pot_timetrace), 'g-', color = 'k') 
    plt.plot(potential_array, average_ratio_1,'o', color = 'k', label='Average midpoint timetrace')
    plt.plot(potential_array, nernst(potential_array, av_pot_FCS), 'g-', color = 'b') 
    plt.plot(potential_array, average_ratio_FCS,'x', color = 'b', label='Average midpoint FCS')
    plt.legend()

    print('The (average) midpoint potential is according to timetrace/FCS: %s'%av_pot_timetrace)
    print('using %s different points' %len(midpoint_potential_array))
    print('The (average) midpoint potential is according to FCS: %s'%av_pot_FCS)
    print('using %s different points' %len(midpoint_potential_array_FCS))

    wb.save(save_filename)

    return()    