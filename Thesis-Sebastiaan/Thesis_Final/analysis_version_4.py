#Import modules
import numpy as np
import pandas as pd
import os.path
from pylab import *
import glob
import os
import re
from xlwt import Workbook
from matplotlib.colors import LogNorm



from numpy import sqrt, pi, exp, linspace, loadtxt
from lmfit import  Model, Parameter, Parameters

import matplotlib.pyplot as plt

wb = Workbook()
sheet1 = wb.add_sheet('on_times',cell_overwrite_ok=True)
sheet2 = wb.add_sheet('off_times',cell_overwrite_ok=True)
sheet1.write(0,0,'Potential (mV):')
sheet2.write(0,0,'Potential (mV):')
sheet1.write(1,0,'Point #')
sheet2.write(1,0,'Point #')
for i in list(range(1,32)):
    sheet1.write(i+2,0,i)
    sheet2.write(i+2,0,i)



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

def average_on_and_off_times(titel, pot, pointnumbers):
    potential, pntnumbers = pot, pointnumbers #creates an array with dimension potential (col) x pntnumbers (rows) 
    on_time = [[None] * potential for i in range(pntnumbers)]
    off_time = [[None] * potential for i in range(pntnumbers)]
    potential_array = []
    all_on_times_array = [[]]
    all_off_times_array = [[]]
    df3 = pd.DataFrame(columns=potential_array)
    df3_off = pd.DataFrame(columns=potential_array)
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(".datn")]:
            number1 = filename[-11:-10] #first number
            number2 = filename[-10:-9] #second number
            if number1.isdigit(): #check if 1 or 2-digit number
                pointnumber = number1 + number2
            elif number2.isdigit():
                pointnumber = number2
            pointnumber = int(pointnumber)  #reading the pointnumber
            string_1 = 'mV'
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
                    os.chdir(dirpath)
                    f_datn = filename
                    f_emplot = re.sub('.datn$','.datn.em.plot',f_datn)
                    if os.path.isfile(f_emplot):               
                        t_on = T_off_average(f_datn, f_emplot)[0]
                        t_off = T_off_average(f_datn, f_emplot)[1]
                        t_on_hist = T_off_average(f_datn, f_emplot)[2]
                        t_off_hist = T_off_average(f_datn, f_emplot)[3]
                        sheet1.write(pointnumber+2,i+1,t_on)
                        sheet2.write(pointnumber+2,i+1,t_off)
                        on_time[pointnumber-1][i] = t_on
                        off_time[pointnumber-1][i] = t_off
                        on_time_df = pd.DataFrame(t_on_hist,columns = [potential_array[i]])
                        off_time_df = pd.DataFrame(t_off_hist,columns = [potential_array[i]])
                        df3 = df3.append(on_time_df)
                        df3_off = df3_off.append(off_time_df)
                    else:
                        print("The file %s does not exist" %f_emplot)
                    os.chdir('..')
                        
    for i in range(len(potential_array)):
        sheet1.write(0,i+1,int(potential_array[i]))
        sheet2.write(0,i+1,int(potential_array[i]))             
        #on_time[0][i] = int(potential_array[i])
        #off_time[0][i] = int(potential_array[i])
    df_ontime = pd.DataFrame(data = on_time, index = range(1,pntnumbers+1), columns = potential_array)
    df_offtime = pd.DataFrame(data = off_time, index = range(1,pntnumbers+1), columns = potential_array)
    av_on = np.transpose(df_ontime.mean()) 
    av_off = np.transpose(df_offtime.mean())
    df_on_shifted = df3.shift(+1) ## shift up
    df_on_shifted.drop(df3.shape[0] - 1,inplace = True)
    df_off_shifted = df3_off.shift(+1) ## shift up
    df_off_shifted.drop(df3_off.shape[0] - 1,inplace = True)
    for i in potential_array:
        fig1 = plt.figure(figsize=(12,10))
        ax1 = fig1.add_subplot(2,2,1)
        hist(df3[i], range=(0,0.5),bins=100)
        ax1.set_title('ON time %smV' %i)
        ax2 = fig1.add_subplot(2,2,2)
        hist(df3_off[i], range=(0,0.5),bins=100)
        ax2.set_title('OFF time %smV' %i)
        ax3 = fig1.add_subplot(2,2,3)
        hist2d(df3[i],df_on_shifted[i], range=[[0,0.25], [0,0.25]],bins=100, norm=LogNorm());
        ax3.set_title('ON time %smV' %i)
        ax3.set_xlabel('Ton/s', fontsize=14, fontname='Times New Roman')
        ax3.set_ylabel('Ton+1/s', fontsize=14, fontname='Times New Roman')
        ax4 = fig1.add_subplot(2,2,4)
        hist2d(df3_off[i],df_off_shifted[i], range=[[0,0.25], [0,0.25]],bins=100, norm=LogNorm());
        ax4.set_title('ON time %smV' %i)
        ax4.set_xlabel('Ton/s', fontsize=14, fontname='Times New Roman')
        ax4.set_ylabel('Ton+1/s', fontsize=14, fontname='Times New Roman')
        plt.tight_layout()
        

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    plt.title('On time') 
    plt.xlabel('potential(mV)')
    plt.ylabel('ms')
    for i in range(len(df_ontime.index)):
        plt.scatter(list(df_ontime),df_ontime.iloc[i])
    
    ax2 = fig.add_subplot(212)
    plt.title('Off time')
    plt.xlabel('potential(mV)')
    plt.ylabel('ms')
    for i in range(len(df_offtime.index)):
        plt.scatter(list(df_offtime),df_offtime.iloc[i])
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    plt.title('Average On time')
    plt.xlabel('potential(mV)')
    plt.ylabel('ms')
    plt.scatter(list(df_ontime),av_on)
    
    ax1 = fig.add_subplot(212)
    plt.title('Average Off time') 
    plt.xlabel('potential(mV)')
    plt.ylabel('ms')
    plt.scatter(list(df_ontime),av_off)
   
    for i in range(len(potential_array)):
        sheet1.write(0,i+1,int(potential_array[i]))
        sheet2.write(0,i+1,int(potential_array[i]))
    wb.save(titel)
    return()    
    