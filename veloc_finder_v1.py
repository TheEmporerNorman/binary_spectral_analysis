# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 20:12:37 2017

@author: Michael
"""
import numpy as np
import matplotlib.pyplot as plt

""" ~~~~~~~~~~ Functions ~~~~~~~~~~ """

def formatGraph(g_title,x_title,y_title):
    
    """ Ease of life function for graph formating """
    
    plt.title( g_title ) 
    plt.xlabel( x_title ) 
    plt.legend(loc='best')
    plt.ylabel(y_title) 
    plt.grid()

def sinCurve(gamma, kx, ky, phase):
    return gamma + kx*np.sin(2*np.pi*phase) + ky*np.cos(2*np.pi*phase)

def sigmaErr(gamma, kx, ky, phase, veloc, N):
          return np.sqrt((1./(N-3.))*np.sum((veloc - sinCurve(gamma, kx, ky, phase))**2))
          
def calcStlrMass(prd,rad_veloc,rad_veloc_err,grav_cnst):
    mass = (prd*rad_veloc**3)/((2*np.pi*grav_cnst))
    err = ((3*prd*(rad_veloc**2))/(2*np.pi*grav_cnst))*rad_veloc_err
    return mass, err
    
""" ~~~~~~~~~~ Variables ~~~~~~~~~~ """

grav_cnst = 6.67508E-11

prd = 0.3440915*86400
prd_err = 0.01

sol_mass = 1.988E30

""" ~~~~~~~~~~ Calculations ~~~~~~~~~~ """

chi_wgts = np.load("chi_wgts.npy")
chi_min_veloc = np.load("chi_min_veloc.npy")
org_phse = np.load("org_phse.npy")
chi_err = np.load("chi_err.npy")

num_spcts = len(chi_min_veloc[0])
num_temps = 1

s = np.arange(num_spcts); t = np.arange(num_temps)

rad_phse = 2*np.pi*org_phse

#~~~~~~~~~~~~ Question 4 ~~~~~~~~~~~~#

s_val = np.sin(rad_phse); c_val = np.cos(rad_phse); w_veloc = chi_min_veloc*chi_wgts

A = np.sum(chi_wgts), np.sum(chi_wgts*s_val), np.sum(chi_wgts*c_val)
B = np.sum(chi_wgts*s_val), np.sum(chi_wgts*s_val**2), np.sum(chi_wgts*s_val*c_val)
C = np.sum(chi_wgts*c_val), np.sum(chi_wgts*c_val*s_val), np.sum(chi_wgts*c_val**2)

RHS = np.sum(w_veloc), np.sum(w_veloc*s_val), np.sum(w_veloc*c_val)
LHS = np.array([A,B,C])

gamma,kx,ky = np.linalg.solve(LHS, RHS)

eig = np.linalg.eig(LHS)

err_mat = np.diag(eig[0])
inv = np.linalg.inv(err_mat)

gamma_err = np.sqrt(inv[0][0])
kx_err = np.sqrt(inv[1][1])
ky_err = np.sqrt(inv[2][2])

arr_phse = np.linspace(min(org_phse), max(org_phse), 1000)
sin_fit = sinCurve(gamma, kx, ky, arr_phse)

fit_err = sigmaErr(gamma,kx,ky,org_phse,chi_min_veloc, num_spcts)

print("Fit Error:", '{0:.3f}'.format(fit_err))

print("Gamma:", '{0:.3f}'.format(gamma), "+-", '{0:.3f}'.format(gamma_err))
print("kx:", '{0:.3f}'.format(ky), "+-", '{0:.3f}'.format(kx_err), "ms-1")
print("ky:", '{0:.3f}'.format(gamma), "+-", '{0:.3f}'.format(ky_err), "ms-1")

for temp_idx in t:
    plt.figure("m0")
    plt.plot(org_phse, chi_min_veloc[temp_idx], "x", label = "Chi minimised velocity")
    plt.errorbar(org_phse, chi_min_veloc[temp_idx], chi_err[temp_idx], fmt = "None", label = "Chi minimised velocity error")
    plt.plot(arr_phse, sin_fit, label = "Fitted sin cuve")
    plt.savefig("m0_veloc")

    formatGraph("","Phaze","Velocity (ms-1)")

#~~~~~~~~~~~~ Question 5 ~~~~~~~~~~~~# b

rad_veloc = np.hypot(kx,ky)
rad_veloc_err = np.hypot(((ky/rad_veloc)*ky_err),((kx/rad_veloc)*kx_err))

print("Radial Velocity:", '{0:.3f}'.format(rad_veloc/1000), "+-", '{0:.3f}'.format(rad_veloc_err/1000), "kms-1")

#~~~~~~~~~~~~ Question 6 ~~~~~~~~~~~~#

mass_fn, mass_fn_err = calcStlrMass(prd, rad_veloc, rad_veloc_err, grav_cnst)
mass_fn_s = mass_fn/sol_mass; mass_fn_err_s = mass_fn_err/sol_mass

print("Mass Function:", '{0:.3f}'.format(mass_fn_s), "+-", '{0:.3f}'.format(mass_fn_err_s), "solar masses")

#~~~~~~~~~~~~ Question 7 ~~~~~~~~~~~~#

mass_c = np.mean([0.45,0.8])
mass_c_err = 0.8 - mass_c

def calcMass(mass_fn, mass_c, i):
    coeff = np.array([np.sin(i)**3, -mass_fn, -2*mass_fn*mass_c, -mass_fn*mass_c**2])
    roots = np.roots(coeff)
    return np.real(roots[np.isreal(roots)])[0]
                   
mass_x = calcMass(mass_fn_s, mass_c, np.pi/2)
mass_err_lrg = calcMass((mass_fn_s + mass_fn_err_s), mass_c, np.pi/2)
mass_err_sml = calcMass((mass_fn_s - mass_fn_err_s), mass_c, np.pi/2)

mass_x_err = np.max([abs(mass_x - mass_err_lrg), abs(mass_x - mass_err_sml)])

print("Mass Object:", '{0:.3f}'.format(mass_x), "+-", '{0:.3f}'.format(mass_x_err), "solar masses")
    
#~~~~~~~~~~~~ Question 8 ~~~~~~~~~~~~#

num_smpls = 10000

i_tri = np.random.normal(np.deg2rad(48), np.deg2rad(11), num_smpls)

#uniform(0.2,np.pi - 0.2, num_smpls)
mc_tri = np.random.normal(mass_c,mass_c_err, num_smpls)
rad_veloc_tri = np.random.normal(rad_veloc, rad_veloc_err, num_smpls)
mass_fn_tri, mass_fn_err_arr = calcStlrMass(prd, rad_veloc_tri, rad_veloc_err, grav_cnst)

smpl_mass_x = np.zeros(len(i_tri))
for smpl_idx in np.arange(len(i_tri)):
    smpl_mass_x[smpl_idx] = calcMass(mass_fn_tri[smpl_idx]/sol_mass,mc_tri[smpl_idx],i_tri[smpl_idx])

mean = np.mean(smpl_mass_x)
med = np.median(smpl_mass_x)
std = np.std(smpl_mass_x)    
    
plt.figure()
plt.hist(smpl_mass_x, bins = 2000, normed = True)
formatGraph("","Mass (solar masses)", "Normalised prrobability")
plt.xlim(0,40)

plt.figure()
plt.hist(smpl_mass_x, bins = 2000, cumulative = True, normed = True)
hist, bins = np.histogram(smpl_mass_x, bins = 10000, normed = True)
plt.xlim(0,40)
formatGraph("","Mass (solar masses)", "Cumulative prrobability")


normal_x = np.linspace(0,40,1000)




