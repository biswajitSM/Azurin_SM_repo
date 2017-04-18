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



def FCS_mono_fit(filename,tmin,tmax):
    df_fcs = pd.read_csv(filename, index_col=False, names=None, skiprows=1, header=None, sep='\ ', engine='python');
    df_fcs = df_fcs[df_fcs[0]>=tmin];
    df_fcs = df_fcs[df_fcs[0]<=tmax];
    df_fcs = df_fcs[np.isfinite(df_fcs[2])]
    np.isnan(df_fcs[0]);#removing nan file
    xdata=df_fcs[0];
    ydata=df_fcs[2];
    def mono_exp(x, A1, A2, t1):
        return((A1+A2*exp(-x/t1)))
    
    monofit, pcov_mono = curve_fit(mono_exp, xdata, ydata, p0 = [10, 1, 1], bounds=(0, np.inf))
    return(monofit)

def FCS_bi_fit(filename,tmin,tmax):
    df_fcs = pd.read_csv(filename, index_col=False, names=None, skiprows=1, header=None, sep='\ ', engine='python');
    df_fcs = df_fcs[df_fcs[0]>=tmin];
    df_fcs = df_fcs[df_fcs[0]<=tmax];
    df_fcs = df_fcs[np.isfinite(df_fcs[2])]
    np.isnan(df_fcs[0]);#removing nan file
    xdata=df_fcs[0];
    ydata=df_fcs[2];
    def biexp(x, A1, A2, t1, A3, t2):
        return(A1+A2*exp(-x/t1))+A3*exp(-x/t2)
    try:
        bifit, pcov_bi = curve_fit(biexp, xdata, ydata, p0 = [10, 1, 1, 0.5, 1], bounds=(0, np.inf))
    except RuntimeError:
        bifit = [NaN,NaN,NaN,NaN,NaN]
        print('Runtime Error %s' %filename)
    return(bifit)

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
                            var2 = variables_fit[1] #B1
                            var4 = variables_fit[3] #B2
                            var5 = variables_fit[2] #C1
                            var6 = variables_fit[4] #C2
                            
                            ton_1 = ((var1/var2) + 1) * var5 
                            ton_2 = ((var1/var4) + 1) * var6
                            ratio_1 = var2/var1
                            ratio_2 = var4/var1
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
                            variables_fit_mono = FCS_mono_fit(filename,tminFCS,tmaxFCS)
                            var1 = variables_fit_mono[0] #A
                            var2 = variables_fit_mono[1] #B
                            ratio_1 = var2 / var1
                            #ratio_test = toff_1 /ton_1
                            #print('Voor pointnummer %s in potentiaal %s is de ratio %s' %(pointnumberFCS, potential_FCS, ratio_1))
                            t_ratio_FCS[pointnumberFCS-1][j] = ratio_1
                            sheet2.write(pointnumberFCS+2,j+1,ratio_1)




                            
                            

                                
    
                  
                        
                        os.chdir(homedir1)
    os.chdir(homedir1)
    os.chdir('..')

    for i in list(range(1,pointnumbers+1)):
        sheet1.write(i+2,0,i)
        sheet2.write(i+2,0,i)

    for i in range(len(potential_array)):
        sheet1.write(0,1+i,int(potential_array[i]))
        sheet2.write(0,1+i,int(potential_array[i]))
    new_array = []
    new_array[:] = [x / 1000 for x in potential_array]

    df_t_ratio = pd.DataFrame(data = t_ratio, index = range(1,pntnumbers+1), columns = new_array)
    df_t_ratio_FCS = pd.DataFrame(data = t_ratio_FCS, index = range(1,pntnumbers+1), columns = new_array)

    

    
    
    #df_t_ratio.drop(0.05, axis=1, inplace=True) #only for the day where 50 mV was not good data
    #df_t_ratio_FCS.drop(0.05, axis=1, inplace=True) #only for the day where 50 mV was not good data
    #new_array.remove(0.05)
    
    #potential_array.remove(50)
    average_ratio_1 =[]
    average_ratio_FCS = []
    #for i in new_array:
    #    average_ratio_1.append(df_t_ratio[i].mean())
    #    average_ratio_FCS.append(df_t_ratio_FCS[i].mean())
    
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
    print('These are the midpoint potentials according to timetrace/FCS:')
    print(midpoint_potential_array)
    
    av_pot_timetrace = sum(midpoint_potential_array)/len(midpoint_potential_array)
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
        #print(list_1_FCS)
        #print(potential_1_FCS)
        del list_1_FCS[:]
        del potential_1_FCS[:]
    
    print('These are the midpoint potentials according to FCS only:')
    print(midpoint_potential_array_FCS)
    av_pot_FCS = sum(midpoint_potential_array_FCS)/len(midpoint_potential_array_FCS)   


    '''
    
    popt_0, pcov_0 = curve_fit(nernst, potential_array, average_ratio_1, p0 = 0.020, bounds = (0,np.inf))
    popt_FCS, pcov_FCS = curve_fit(nernst, potential_array, average_ratio_FCS, p0 = 0.020, bounds = (0,np.inf))
    
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
    '''

    print('The (average) midpoint potential is according to timetrace/FCS: %s'%av_pot_timetrace)
    print('using %s different points' %len(midpoint_potential_array))
    print('The (average) midpoint potential is according to FCS: %s'%av_pot_FCS)
    print('using %s different points' %len(midpoint_potential_array_FCS))
    #day 1 10points S101d16Feb17_60.5_635_A3_CuAzu655
    '''day_1_TT = [-0.012445944137990751, 0.0067739895407066949, -0.0017496319108184608, -0.0024372534638395526, -0.0082462800543958818, -0.013318143829450332, -0.021205432715052021, -0.032243390976523199, -0.0061805678648803181, 0.0013353258709235866]
    day_1_FCS = [-0.02731137284739002, -0.01838188782651785, 0.0079568102231238492, -0.0096676584084904905, 0.0018928855955552097, 0.0073000060380700579, 0.0089745869019659235, 0.011524762485761045, -0.0085245238003953724, -0.0081424155457395179]
#day 2 17points S104d20Feb17_60.5_635_A1_CuAzu655
    day_2_TT = [-0.015115695689253892, -0.0014817851017409529, -0.0015407426984148287, -0.0043248720551140639, -0.0045718278449559365, 0.018482805319311332, 0.0071592377269750511, -0.0059294763118028729, -0.0099647021891172832, -0.0018850608078926352, 0.031683395182030193, 0.0060502384041959522, 0.026018264000545435, 0.013609627587096482, 0.027865569058411574, -5.5301774458483778e-05, 0.028971503395359741]
    day_2_FCS = [0.0071059626829535205, -0.00076558284753633983, 0.013088760446317113, -0.005162666731122984, 0.0019905549451677321, 0.03239198451766119, -0.0049898808746936894, 0.0017387725725446746, 0.02481005339820128, -0.0018052102172336746, 0.0027723497740089432, 0.0078739792484701712, 0.073310075073377792, 0.021731305046203794, 0.0089217729753382335, -0.0053948328896486451, 0.030199800275347889]
#day 3 10 points S101d16Feb17_60.5_635_A3_CuAzu655
    day_3_TT = [-0.012297715155593856, 0.0034791538692711841, -0.0011107451089841984, 0.00072037560693665945, -0.008504546373254155, -0.017112386012455752, -0.020365974588592892, -0.0321114996920423, -0.0057746142462300433, 0.0033897873607004205]
    day_3_FCS = [-0.027228497356434671, -0.020921941713098333, 0.0088953654085984837, -0.0071415884808491955, 0.001510181469115808, 0.0052295072986914895, 0.011807381976336403, 0.012261238413476889, -0.0081607686817894987, -0.006740939257021651]
#day 4 17 points S101d15Feb17_62.2_635_A2_CuAzu655_2nd
    day_4_TT = [-0.019795855497937487, -0.0020926151291406399, 0.012753177965378603, 0.0024349928356864523, 0.0065341725980482468, -0.019526254710581569, 0.00059715105494523703, -0.0069342555834307961, -0.0015910370362179835, -0.015770837203604011, -0.012197238380629551, 0.00045512890907429838, -0.0041433007308348101, -0.060525066055497706, -0.00654357390136758, -0.018058473238781091, -0.0036982785580528506]
    day_4_FCS = [-0.0096889295374154097, 0.0018559142877343688, -0.0056755779750995794, 0.016681064291581282, 0.0012358966375751942, 0.0016779899552443202, -0.0035086887337545673, 0.0081878981285823903, -0.0063558919369876999, -0.0072256365985315868, 0.0027555521635530414, -0.0017088688920240641, -0.026478745389741064, 0.0041595097057160859, -0.0019787021095678172, 0.0069577559222336865, -0.011713461839314885]
#day 5 11 points S101d14Feb17_60.5_635_A2_CuAzu655
    day_5_TT = [-0.011201328709071662, -0.014717294920857077, 0.0040800403161013526, 0.0082447426816030442, 0.0020957539624569853, 0.0036996365596383574, -0.0061547744736242622, -0.0047020319816483526, -0.009197346666088747, 0.014387231931035261, 0.011768518447300307]
    day_5_FCS = [-0.010044507475404977, 0.014889941122317082, 0.0092419138163808388, 0.032947658816004859, 0.0016912384941170024, -0.0090899030419872937, 0.012001995934161637, 0.00028284856411316524, 0.01753217497763641, 0.0051159103501601947, -0.0018602628361905859]
    point_1_y_FCS = [17.150298331939851, 1.9280799721143982, 6.3851976099582677, 100.23543367298997, 36.551482559770236]
    point_1_x = [0.0, 0.025, 0.035, 0.05, 0.1]
    point_1_y_TT = [17.150298331939851, 1.9280799721143982, 6.3851976099582677, 26.538838934170819, 24.39714017142753]
    point_2_y_TT = [8.9552710058767477, 10.864962694285907, 3.6480918854052415, 68.270333733574731, 4.0690606660090038, 5.8626950591620508]
    point_2_x = [0.0, 0.025, 0.06, 0.1, 0.01, 0.045]
    point_2_y_FCS = [8.9552710058767477, 10.864962694285907, 4.0936193573557995, 3.4654133390982302, 18.968781113328454, 4.0690606660090038]
    point_3_y_FCS = [8.2588059539418968, 8.8428954559351372, 18.146037851646891, 20.540035526603983]
    point_3_x = [0.0, 0.025, 0.075, 0.1]
    point_3_y_TT = [8.2588059539418968, 8.8428954559351372, 24.34092151747085, 87.237190895678154]
    combined_TT = day_1_TT + day_2_TT + day_3_TT + day_4_TT + day_5_TT
    combined_FCS = day_1_FCS + day_2_FCS + day_3_FCS + day_4_FCS + day_5_FCS
    av_TT = sum(combined_TT)/len(combined_TT)
    av_FCS = sum(combined_FCS)/len(combined_FCS)

    plt.figure(figsize = (10,10))
    ax1 = plt.subplot2grid((2,3), (0,0), rowspan = 2, colspan = 2)
    ax1.set_yscale('log')
    ax1.set_xlim(-0.01, 0.11)
    ax1.plot(potential_array, nernst(potential_array, av_TT), 'g-', color = 'c', linewidth = 3, label = "Timetrace") #TT
    ax1.plot(point_1_x, point_1_y_TT, 'x', color = 'r')
    ax1.plot(point_1_x, point_1_y_FCS, 'o', color = 'r')
    ax1.plot(point_2_x, point_2_y_TT, 'x', color = 'y')
    ax1.plot(point_2_x, point_2_y_FCS, 'o', color = 'y')
    ax1.plot(point_3_x, point_3_y_TT, 'x', color = 'g')
    ax1.plot(point_3_x, point_3_y_FCS, 'o', color = 'g')

    #ax1.plot(potential_array, average_ratio_1,'o', color = 'k', label='Average midpoint potential timetrace')
    ax1.plot(potential_array, nernst(potential_array, av_FCS), 'g-', color = 'b', linewidth = 3, label = "FCS") 
    #ax1.plot(potential_array, average_ratio_FCS,'x', color = 'b', label='Average midpoint FCS')
    ax1.legend()
    
   
    
    
    ax2 = plt.subplot2grid((2,3), (0,2))
    
    n,bins_on1,patches = hist(combined_TT, bins = 10)
    ax2.hist(combined_TT, bins = 10, color = 'c')
    bin_centers_on = bins_on1[:-1] + 0.5 * (bins_on1[1:] - bins_on1[:-1])
    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))


    popt_single, pcov_single = curve_fit(gaus, bin_centers_on, n)
    ax2.plot(bin_centers_on,gaus(bin_centers_on,*popt_single), color = 'r', linewidth = 3)
    ax3 = plt.subplot2grid((2,3), (1,2), colspan = 1)
    ax3.hist(combined_FCS, bins = 10, color = 'b')

    n,bins_off1,patches = hist(combined_FCS, bins = 10)

    bin_centers_off = bins_off1[:-1] + 0.5 * (bins_off1[1:] - bins_off1[:-1])
    popt_FCS, pcov_FCS = curve_fit(gaus, bin_centers_off, n)

    ax3.plot(bin_centers_off,gaus(bin_centers_off,*popt_FCS), color = 'r', linewidth = 3)

    ax2.yaxis.set_label_position("right")
    ax3.yaxis.set_label_position("right")

    ax1.set_xlabel('Potential (mV)', fontsize=15, fontname='Times New Roman')
    ax3.set_xlabel('Potential (mV)', fontsize=15, fontname='Times New Roman')
    ax3.set_ylabel('#', fontsize=15, fontname='Times New Roman')
    ax2.set_ylabel('#', fontsize=15, fontname='Times New Roman')
    ax1.set_ylabel(r'$\tau_{off}\tau_{on}^{-1}$')
    #ax2.set_xticks([-0.03, 0, 0.03])
    #ax3.set_xticks([0.02,0.04,0.06,0.08])
    wb.save(save_filename)
    plt.tight_layout()
    plt.show()
    '''
    return()    