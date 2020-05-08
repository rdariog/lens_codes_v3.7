import sys
import numpy as np
from pylab import *
from multipoles_shear import *
import emcee
import time
from multiprocessing import Pool
import argparse
from astropy.io import fits
from astropy.cosmology import LambdaCDM
from profiles_fit import *
#parameters

cvel = 299792458;   # Speed of light (m.s-1)
G    = 6.670e-11;   # Gravitational constant (m3.kg-1.s-2)
pc   = 3.085678e16; # 1 pc (m)
Msun = 1.989e30 # Solar mass (kg)


parser = argparse.ArgumentParser()
parser.add_argument('-folder', action='store', dest='folder',default='./')
parser.add_argument('-file', action='store', dest='file_name', default='profile.cat')
parser.add_argument('-ncores', action='store', dest='ncores', default=4)
args = parser.parse_args()

folder    = args.folder
file_name = args.file_name
ncores    = args.ncores
ncores    = int(ncores)

print('fitting monopole misscentred')
print(folder)
print(file_name)

profile = fits.open(folder+file_name)
h       = profile[1].header
p       = profile[1].data
zmean   = h['Z_MEAN']    
Mhalo   = 10**h['lMASS_HALO_mean']
Rmean   = h['RADIUS_HALO_mean']
ROUT = (2.5*(2.*(Mhalo/2.21e14)**0.75)**(1./3.))/0.7
soff = 0.4*Rmean

# Compute cosmological parameters
cosmo = LambdaCDM(H0=0.7*100, Om0=0.3, Ode0=0.7)
H        = cosmo.H(zmean).value/(1.0e3*pc) #H at z_pair s-1 
roc      = (3.0*(H**2.0))/(8.0*np.pi*G) #critical density at z_pair (kg.m-3)
roc_mpc  = roc*((pc*1.0e6)**3.0)


def log_likelihood(data_model, r, Gamma, e_Gamma):
    log_M200,pcc = data_model
    M200 = 10**log_M200
    multipoles = multipole_shear_parallel(r,M200=M200,
                                misscentred = True,s_off = soff,
                                ellip=0,z=zmean,components = ['t'],
                                verbose=False,ncores=ncores)
    model = model_Gamma(multipoles,'t', misscentred = True, pcc = pcc)
    sigma2 = e_Gamma**2
    return -0.5 * np.sum((Gamma - model)**2 / sigma2 + np.log(2.*np.pi*sigma2))
    

def log_probability(data_model, r, Gamma, e_Gamma):
    log_M200, pcc = data_model
    if 11. < log_M200 < 15.5 and 0.3 < pcc < 1.0:
        return log_likelihood(data_model, r, Gamma, e_Gamma)
    return -np.inf

# initializing

pos = np.array([np.random.uniform(11.5,15.0,10),
                np.random.uniform(0.3,0.8,10)]).T


nwalkers, ndim = pos.shape

#-------------------
# running emcee

#pool = Pool(processes=(ncores))

maskr = (p.Rp < ROUT)

t1 = time.time()
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                args=(p.Rp[maskr],p.DSigma_T[maskr],p.error_DSigma_T[maskr]))
sampler.run_mcmc(pos, 300, progress=True)
print('//////////////////////')
print('         TIME         ')
print('----------------------')
print((time.time()-t1)/60.)
#pool.terminate()

#-------------------
# saving mcmc out

mcmc_out = sampler.get_chain(flat=True)

f1=open(folder+'monopole_pcconly_'+file_name[:-4]+'out','w')
f1.write('# log(M200)  pcc  \n')
np.savetxt(f1,mcmc_out,fmt = ['%12.6f']*2)
f1.close()
