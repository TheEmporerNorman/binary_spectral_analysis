# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:02:58 2017

@author: Michael
"""
import numpy as np 
import scipy.interpolate as it
import matplotlib.pyplot as plt
import os

""" ~~~~~~~~~~ Functions ~~~~~~~~~~ """


def formatGraph(g_title,x_title,y_title):
    
    """ Ease of life function for graph formating """
    
    plt.title( g_title ) 
    plt.xlabel( x_title ) 
    plt.legend(loc='best')
    plt.ylabel(y_title) 
    plt.grid()

def readSpct(spect):
    
    """ Reads spectrograms from inputted folder into dictionary"""
    
    num_files = 0
    
    for file in os.listdir(spect.dir_name):
        if file.endswith(spect.ext):
            num_files += 1 
            
    file_names = [None]*num_files
    
    file_idx = 0
    
    x_data = [None]*num_files
    y_data = [None]*num_files
    y_err = [None]*num_files
    
    for file in os.listdir(spect.dir_name):
        if file.endswith(spect.ext):
            file_name = file.rstrip(spect.ext)
            file_names[file_idx] = file_name
            x_data[file_idx], y_data[file_idx], y_err[file_idx] = np.loadtxt(os.path.join(spect.dir_name, file), unpack = True)
            file_idx += 1

    spect.num = num_files
    spect.names = file_names
    spect.raw_x = np.asarray(x_data)
    spect.raw_y = np.asarray(y_data)
    spect.err_y = np.asarray(y_err)
            
    return spect
    
def convAngToMet(spect):
    
    spect.conv_x = np.zeros_like(spect.raw_x)
    
    n = np.arange(spect.num)
    for data_idx in n:
        spect.conv_x[data_idx]= spect.raw_x[data_idx]*1E-10
    return spect
    
def calcMovAvg(data, err, wind_size):
    
    """ Calculates weighted moving average given data set with errors given """
    
    wghts = (1./(err**2.))
    w_data = data*wghts
    
    cumsum_data = np.cumsum(np.insert(w_data, 0, 0))
    cumsum_wghts = np.cumsum(np.insert(wghts, 0, 0)) 
    mov_avg = (cumsum_data[wind_size:] - cumsum_data[:-wind_size]) / (cumsum_wghts[wind_size:] - cumsum_wghts[:-wind_size]) 
    pad_mov_avg = np.zeros(wind_size + len(mov_avg) - 1)
    pad_mov_avg[(np.floor(wind_size/2) - 1):(np.floor(-wind_size/2))] = mov_avg

    mov_avg_err = (np.sqrt(1/(cumsum_wghts[wind_size:] - cumsum_wghts[:-wind_size])*np.sum((data - pad_mov_avg))))
    
    pad_mov_avg_err = np.zeros(wind_size + len(mov_avg) - 1)
    pad_mov_avg_err[(np.floor(wind_size/2) - 1):(np.floor(-wind_size/2))] = mov_avg_err
    
    return pad_mov_avg, pad_mov_avg_err

def calcSpctAvg(spect):
    
    """ Calculates moving average for spectrum """

    spect.mov_avg_y = np.zeros_like(spect.raw_y)
    spect.mov_avg_y_err = np.zeros_like(spect.raw_y)
    
    n = np.arange(spect.num)
    for data_idx in n:    
        spect.mov_avg_y[data_idx], spect.mov_avg_y_err[data_idx] = calcMovAvg(spect.raw_y[data_idx],spect.err_y[data_idx],spect.wind_size)
    return spect
    
def calcSpctSpln(spect):
    
    spect.spln_y = np.zeros_like(spect.raw_y)
    spect.diff_y = np.zeros_like(spect.raw_y)
    spect.spln_err_y = np.zeros(spect.num)

    """ Calculates moving average for spectrum """
    
    n = np.arange(spect.num)
    for data_idx in n:
        spln_func = it.UnivariateSpline(spect.conv_x[data_idx][spect.wind_size/2:(-spect.wind_size/2) - 1:spect.spln_size], spect.mov_avg_y[data_idx][spect.wind_size/2:(-spect.wind_size/2) - 1:spect.spln_size], w = spect.mov_avg_y_err[data_idx][spect.wind_size/2:(-spect.wind_size/2) - 1:spect.spln_size] )
        spect.spln_err_y[data_idx] = np.sqrt(1/len(spect.mov_avg_y[data_idx][spect.wind_size/2:(-spect.wind_size/2) - 1:spect.spln_size])*spln_func.get_residual())
        spect.spln_y[data_idx] = spln_func(spect.conv_x[data_idx])
        spect.diff_y[data_idx] = spect.raw_y[data_idx] - spect.spln_y[data_idx]       
    return spect

def calcDopShft(data,veloc,c_veloc):
    return data*(c_veloc+veloc)/c_veloc
    
def calcSpctShft(temps, spcts, velocs, c_veloc, num_veloc, det):
    temps.shft_x = np.zeros((temps.num, num_velocs, len(temps.conv_x[0])))
    temps.int_fn =  np.zeros((temps.num, num_velocs))
    temps.itrp_x = np.zeros((temps.num,det)) 
    temps.itrp_y = np.zeros((temps.num, num_velocs, det))
    
    spcts.itrp_y = np.zeros((temps.num, spcts.num, num_velocs, len(spcts.conv_x[0]))) 

    v = np.arange(num_velocs)
    t = np.arange(temps.num)
    s = np.arange(spcts.num)
    
    for temp_idx in t:
        for spct_idx in s:
            temps.itrp_x[temp_idx] = np.linspace(temps.conv_x[temp_idx][0] - 0.5*(temps.conv_x[temp_idx][-1] - temps.conv_x[temp_idx][0]), temps.conv_x[temp_idx][-1] + 0.5*(temps.conv_x[temp_idx][-1] - temps.conv_x[temp_idx][0]), det)
            for veloc_idx in v:
                temps.shft_x[temp_idx][veloc_idx] = calcDopShft(temps.conv_x[temp_idx], velocs[veloc_idx], c_veloc)
                spln_func = it.InterpolatedUnivariateSpline(temps.shft_x[temp_idx][veloc_idx], temps.diff_y[temp_idx])
                spcts.itrp_y[temp_idx][spct_idx][veloc_idx] = spln_func(spcts.conv_x[spct_idx])
            
    return temps
    
def calcShftInrp(spcts, num_velocs, inrp_det):
    
    temps.itrp_x = np.zeros((temps.num, inrp_det)) 
    temps.itrp_y = np.zeros((temps.num, num_velocs, inrp_det))
    
    spcts.itrp_y = np.zeros((temps.num, spcts.num, num_velocs, len(spcts.conv_x[0]))) 
    
    t = np.arange(temps.num)
    s = np.arange(spcts.num)
    
    for temp_idx in t:
        for spct_idx in s:
            temps.itrp_x[temp_idx] = np.linspace(temps.conv_x[temp_idx][0] - 0.5*(temps.conv_x[temp_idx][-1] - temps.conv_x[temp_idx][0]), temps.conv_x[temp_idx][-1] + 0.5*(temps.conv_x[temp_idx][-1] - temps.conv_x[temp_idx][0]), inrp_det)
       
def plotTemp(temp, num_velocs):
    
    """ PLots spectrograms from dictionary """
    
    t = np.arange(temp.num)
    
    for temp_idx in t:
        plt.figure(temp.names[temp_idx])
        plt.errorbar(temp.conv_x[temp_idx],temp.raw_y[temp_idx], temp.err_y[temp_idx], color= "red", label = "Error bars" )
        plt.plot(temp.conv_x[temp_idx],temp.raw_y[temp_idx], "x", label = "Data Points")
        plt.plot(temp.conv_x[temp_idx],temp.mov_avg_y[temp_idx], label = "Moving Average")
        plt.plot(temp.conv_x[temp_idx],temp.spln_y[temp_idx], label = "Spline")
        formatGraph(temp.names[temp_idx],"Wavelength(m)","Intensity")
        
        plt_rng = temp.wind_size/2

        plt.figure()
        plt.plot(temp.conv_x[temp_idx][plt_rng:-plt_rng],temp.diff_y[temp_idx][plt_rng:-plt_rng], "x", label = "Difference")
        formatGraph(temp.names[temp_idx],"Wavelength(m)","Intensity")

def plotSpct(spect, temp, scle_fcts, num_velocs):
    
    """ PLots spectrograms from dictionary """
    
    s = np.arange(spect.num)
    
    for spct_idx in s:
        plt.figure(spect.names[spct_idx])
        plt.errorbar(spect.conv_x[spct_idx],spect.raw_y[spct_idx], spect.err_y[spct_idx], color= "red", label = "Error bars" )
        plt.plot(spect.conv_x[spct_idx],spect.raw_y[spct_idx], "x", label = "Data Points")
        plt.plot(spect.conv_x[spct_idx],spect.mov_avg_y[spct_idx], label = "Moving Average")
        plt.plot(spect.conv_x[spct_idx],spect.spln_y[spct_idx], label = "Spline")
        formatGraph(spect.names[spct_idx],"Wavelength(m)","Intensity")
        
        plt.figure()
        plt.plot(spect.conv_x[spct_idx],spect.diff_y[spct_idx], "x", label = "Difference")
        formatGraph(spect.names[spct_idx],"Wavelength(m)","Intensity")
    
        #plt.plot(spect.conv_x[data_idx], spect.itrp_y[data_idx][0][268]*scle_fcts[data_idx][0][268])
    
def scleSpct(data, temp, data_err):
    return sum((data*temp/(data_err**2)))/sum((temp)**2/(data_err**2))
    
def chiSqr(y,x,err_y,A):
    return np.sum(((y - (A*x))/err_y)**2)
    
""" ~~~~~~~~~~ Class Setup ~~~~~~~~~~ """

class Spect:
    
    names = ""
    num = []

    dir_name = ""
    ext = ""

    raw_x = []
    conv_x = []
    
    shft_x = []
    
    raw_y = []; mov_avg_y = []; spln_y = []; diff_y = []

    err_y = []; mov_avg_y_err = []; spln_err_y = [];

    wind_size = 0; spln_size = 0
    
    int_fn = []

    itrp_x = []
    itrp_y = []
    
    
spcts = Spect()
temps = Spect()

""" ~~~~~~~~~~ Variables ~~~~~~~~~~ """
    
spcts.dir_name = "./gs2000" #<--- Directory location of spectrum files.
temps.dir_name = "./keck_temp" #<--- Directory location of template files.

grav_cnst = 6.67508E-11

spcts.ext = ".txt"
temps.ext = ".txt"

spcts.wind_size = 200 #<--- GS200 window size.
spcts.spln_size = 300

temps.wind_size = 157 #<--- Template window size.
temps.spln_size = 220

c_veloc = 299792458

min_veloc = -600E3
max_veloc = 600E3

num_velocs = 2000

org_phse = np.array([-0.1405,-0.0583,0.0325,0.0998,0.1740,0.2310,0.3079,0.3699,0.4388,0.5008,0.5698,0.6371,0.7276])

""" ~~~~~~~~~~ Calculations ~~~~~~~~~~ """

spcts = readSpct(spcts)
temps = readSpct(temps)

spcts = convAngToMet(spcts)
temps = convAngToMet(temps)

spcts = calcSpctAvg(spcts)
temps = calcSpctAvg(temps)

spcts = calcSpctSpln(spcts)
temps = calcSpctSpln(temps)

velocs = np.linspace(min_veloc,max_veloc,num_velocs)
temps = calcSpctShft(temps, spcts, velocs, c_veloc, num_velocs, 2000)

s = np.arange(spcts.num)
v = np.arange(num_velocs)
t = np.arange(temps.num)

scle_fcts = np.zeros((temps.num, spcts.num, num_velocs))
chi_sqrd = np.zeros((temps.num, spcts.num, num_velocs))

min_points = np.zeros((temps.num, spcts.num))
chi_min_veloc = np.zeros((temps.num, spcts.num))
chi_err = np.zeros((temps.num, spcts.num))

for temp_idx in t:
    for spct_idx in s:
        print(spct_idx)    
        for veloc_idx in v:
            scle_fcts[temp_idx][spct_idx][veloc_idx] = scleSpct(spcts.diff_y[spct_idx][200:-200], spcts.itrp_y[temp_idx][spct_idx][veloc_idx][200:-200], spcts.err_y[spct_idx][200:-200])
            chi_sqrd[temp_idx][spct_idx][veloc_idx] = chiSqr(spcts.diff_y[spct_idx][200:-200], spcts.itrp_y[temp_idx][spct_idx][veloc_idx][200:-200], spcts.err_y[spct_idx][200:-200], scle_fcts[temp_idx][spct_idx][veloc_idx] )
   
        min_points[temp_idx][spct_idx] = min(chi_sqrd[temp_idx][spct_idx])
        chi_min_veloc[temp_idx][spct_idx] = velocs[np.argmin(chi_sqrd[temp_idx][spct_idx])]
        
        chiFunc = it.InterpolatedUnivariateSpline(velocs, chi_sqrd[temp_idx][spct_idx])
        temp_veloc = np.linspace(min_veloc, max_veloc, 10000)
        temp_chi = chiFunc(temp_veloc)
        chi_err[temp_idx][spct_idx] = np.abs(chi_min_veloc[temp_idx][spct_idx] - max(temp_veloc[np.argwhere(temp_chi < (min_points[temp_idx][spct_idx] + 1))]))

chi_wgts = 1/(chi_err**2)
#plotSpct(spcts, temps, scle_fcts, num_velocs)
#plotTemp(temps, num_velocs)
"""
for temp_idx in t:
    for spct_idx in s:
        #plt.figure()
        #plt.plot(velocs, chi_sqrd[temp_idx][spct_idx])
        #formatGraph(spcts.names[spct_idx], "Velocity(ms-1)", "Chi Squared")
"""

np.save("chi_wgts",chi_wgts)
np.save("chi_min_veloc", chi_min_veloc)
np.save("org_phse", org_phse)
np.save("velocs", velocs)
np.save("chi_err", chi_err)






