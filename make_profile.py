import sys
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from astropy.io import fits
import pylab as pylab
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext

import time

#parameters

cvel = 299792458;   # Speed of light (m.s-1)
G    = 6.670e-11;   # Gravitational constant (m3.kg-1.s-2)
pc   = 3.085678e16; # 1 pc (m)
Msun = 1.989e30 # Solar mass (kg)

def covariance_matrix(array):
	
	nobs    = len(array)
	
	CV = np.zeros((nobs,nobs))
	
	for i in range(nobs):
		nsample = len(array[i])
		mean_ei = np.mean(array[i])
		
		for j in range(nobs):

			mean_ej = np.mean(array[j]) 
			
			CV[i,j] = np.sqrt((abs(array[i] - mean_ei)*abs(array[j] - mean_ej)).sum()/nsample)/np.sqrt(array[i].std()*array[j].std())
	
	return CV

def bootstrap_errors_stack(et,ex,peso,nboot,array):
	unique = np.unique(array)
	with NumpyRNGContext(1):
		bootresult = bootstrap(unique, nboot)
		
	et_means = np.array([np.average(et[np.in1d(array,x)],weights=peso[np.in1d(array,x)]) for x in bootresult])
	ex_means = np.array([np.average(ex[np.in1d(array,x)],weights=peso[np.in1d(array,x)]) for x in bootresult])
	
	return np.std(et_means),np.std(ex_means),et_means,ex_means

def errors_disp_halos(et,ex,peso,array):
	unique = np.unique(array)
		
	et_means = np.array([np.average(et[np.in1d(array,x)],weights=peso[np.in1d(array,x)]) for x in unique])
	ex_means = np.array([np.average(ex[np.in1d(array,x)],weights=peso[np.in1d(array,x)]) for x in unique])
	
	return np.std(et_means)/np.sqrt(len(unique)),np.std(ex_means)/np.sqrt(len(unique)),et_means,ex_means

def bootstrap_errors(et,ex,peso,nboot):
	index=np.arange(len(et))
	with NumpyRNGContext(1):
		bootresult = bootstrap(index, nboot)
	INDEX=bootresult.astype(int)
	ET=et[INDEX]	
	EX=ex[INDEX]	
	W=peso[INDEX]	
	
	et_means=np.average(ET,axis=1,weights=W)
	ex_means=np.average(EX,axis=1,weights=W)
	
	return np.std(et_means),np.std(ex_means),et_means,ex_means

def qbootstrap_errors(et,ex,peso,angle,nboot):
	index=np.arange(len(et))
	with NumpyRNGContext(1):
		bootresult = bootstrap(index, nboot)
	INDEX=bootresult.astype(int)
	ET=et[INDEX]	
	EX=ex[INDEX]	
	W=peso[INDEX]	
	A = angle[INDEX]
	
	et_means = np.sum((ET*np.cos(2.*A)*W),axis=1)/np.sum(((np.cos(2.*A)**2)*W),axis=1)
	ex_means = np.sum((EX*np.sin(2.*A)*W),axis=1)/np.sum(((np.sin(2.*A)**2)*W),axis=1)
	
	return np.std(et_means),np.std(ex_means),et_means,ex_means


def shear_profile_log(RIN,ROUT,r,et,ex,peso,m,sigma_c,
                      ndots=15,stepbin=False,booterror_flag=False,
                      lin=False,boot_stack=[],nboot = 100,cov_matrix = False):
	
	'''
	COMPUTE DENSITY PROFILE
	
	------------------------------------------------------
	INPUT
	------------------------------------------------------
	RIN               (float) Radius in kpc from which it is going 
	                  to start binning
	ROUT              (float) Radius in kpc from which it is going 
	                  to finish binning
	r                 (float array) distance from the centre in kpc
	et                (float array) tangential ellipticity component
	                  scaled by the critical density (M_sun/pc^2)
	ex                (float array) cross ellipticity component
	                  scaled by the critical density (M_sun/pc^2)
	peso              (float array) weight for each shear component
	                  scaled according to sigma_c^-2
	m                 (float array) correction factor
	sigma_c           (float array) critical density (M_sun/pc^2)
	                  used only to compute the error in each bin
	                  considering shape noise only
	ndots             (int) number of bins in the profile
	stepbin           (float) length of the bin instead of ndots
	                  if False, it is going to use ndots
	booterror_flag    (bool) if True it is going to use bootstraping
	                  to compute the error in each bin
	lin               (bool) if True it is going to use linalg spacing
	                  between the bins
	boot_stack        (array) used to do the bootstraping, if it is empty
	                  the bootstrap is going to be executed over the whole
	                  sample
	nboot             (int) number of bootstrap repetitions
	cov_matrix        (bool) if true it is going to compute the covariance matrix
	                  - this is still in testing process

	------------------------------------------------------
	OUTPUT
	------------------------------------------------------


	'''


	
		
	if lin:
		if stepbin:
			nbin = int((ROUT - RIN)/stepbin)
		else:
			nbin = int(ndots)
		bines = np.linspace(RIN,ROUT,num=nbin+1)
	else:
		if stepbin:
			nbin = int((np.log10(ROUT) - np.log10(RIN))/stepbin)
		else:
			nbin = int(ndots)
		bines = np.logspace(np.log10(RIN),np.log10(ROUT),num=nbin+1)
		
	if cov_matrix and len(boot_stack):

		ides       = np.unique(boot_stack)
		digit      = np.digitize(r,bines)
		totbines   = np.arange(1,nbin+1)
		maskid     = np.array([all(np.in1d(totbines,digit[boot_stack==x])==True) for x in ides])
		maskides   = np.in1d(boot_stack,ides[maskid])
		boot_stack = boot_stack[maskides]
		r          = r[maskides]
		et         = et[maskides]
		ex         = ex[maskides]
		peso       = peso[maskides]
		m          = m[maskides]
		sigma_c    = sigma_c[maskides]
		if nboot > maskid.sum() and len(boot_stack):
			nboot = maskid.sum()


	etboot = []
	exboot = []


	SHEAR=np.zeros(nbin,float)
	CERO=np.zeros(nbin,float)
	R=np.zeros(nbin,float)
	err=np.zeros(nbin,float)
	error_et=np.zeros(nbin,float)
	error_ex=np.zeros(nbin,float)
	Mcorr=np.zeros(nbin,float)
	N=np.zeros(nbin,float)
		
	
	for BIN in np.arange(nbin):
		# print 'BIN',BIN
		rin  = bines[BIN]
		rout = bines[BIN+1]
		maskr=(r>=rin)*(r<rout)	
		w2=peso[maskr]
		pes2=w2.sum()			
		shear=et[maskr]
		cero=ex[maskr]
		ERR=((sigma_c[maskr]*w2)**2)
		mcorr=m[maskr]
		n=len(shear)
		R[BIN]=rin+(rout-rin)/2.0	
		#~print n
		N[BIN] = n
		if n == 0:
			SHEAR[BIN]=0.0
			CERO[BIN]=0.0
			err[BIN]=0.
			error_et[BIN],error_ex[BIN]=0.,0.
			Mcorr[BIN]=1.
		else:	
			SHEAR[BIN]=np.average(shear,weights=w2)
			CERO[BIN]=np.average(cero,weights=w2)
			sigma_e=(0.28**2.)
			ERR2=(ERR*sigma_e).sum()
			err[BIN]=((ERR2)/((pes2.sum())**2))**0.5			
			Mcorr[BIN]=1+np.average(mcorr,weights=w2)
			if booterror_flag:
				if len(boot_stack):
					error_et[BIN],error_ex[BIN],etboot0,exboot0 = bootstrap_errors_stack(shear,cero,w2,nboot,boot_stack[maskr])
				else:
					error_et[BIN],error_ex[BIN],etboot0,exboot0 = bootstrap_errors(shear,cero,w2,nboot)
				etboot += [etboot0]
				exboot += [exboot0]
				#SHEAR[BIN]=np.average(etboot0)
				#SHEAR[BIN]=np.average(exboot0)
			elif len(boot_stack):
				error_et[BIN],error_ex[BIN],etboot0,exboot0 = errors_disp_halos(shear,cero,w2,boot_stack[maskr])
				etboot += [etboot0]
				exboot += [exboot0]
			else:
				error_et[BIN],error_ex[BIN] = err[BIN],err[BIN]
		
	if cov_matrix:
		CVet = covariance_matrix(etboot)
		CVex = covariance_matrix(exboot)
	else:
		CVet = None
		CVex = None
		
	return [R,SHEAR/Mcorr,CERO/Mcorr,err/Mcorr,nbin,error_et/Mcorr,error_ex/Mcorr,N,CVet,CVex]


def quadrupole_profile_log(RIN,ROUT,r,et,ex,peso,m,sigma_c,angle,
                      ndots=15,stepbin=False,booterror_flag=False,
                      lin=False,nboot = 100):
		
	if lin:
		if stepbin:
			nbin = int((ROUT - RIN)/stepbin)
		else:
			nbin = int(ndots)
		bines = np.linspace(RIN,ROUT,num=nbin+1)
	else:
		if stepbin:
			nbin = int((np.log10(ROUT) - np.log10(RIN))/stepbin)
		else:
			nbin = int(ndots)
		bines = np.logspace(np.log10(RIN),np.log10(ROUT),num=nbin+1)
		

	etboot = []
	exboot = []


	SHEAR=np.zeros(nbin,float)
	CERO=np.zeros(nbin,float)
	R=np.zeros(nbin,float)
	err=np.zeros(nbin,float)
	error_et=np.zeros(nbin,float)
	error_ex=np.zeros(nbin,float)
	Mcorr=np.zeros(nbin,float)
	N=np.zeros(nbin,float)
		
	
	for BIN in np.arange(nbin):
		# print 'BIN',BIN
		rin    = bines[BIN]
		rout   = bines[BIN+1]
		maskr  = (r>=rin)*(r<rout)	
		w2     = peso[maskr]
		pes2   = w2.sum()			
		shear  = et[maskr]
		cero   = ex[maskr]
		ERR    = ((sigma_c[maskr]*w2)**2)
		mcorr  = m[maskr]
		n      = len(shear)
		R[BIN] = rin+(rout-rin)/2.0	
		#~print n
		N[BIN] = n
		if n == 0:
			SHEAR[BIN]=0.0
			CERO[BIN]=0.0
			err[BIN]=0.
			error_et[BIN],error_ex[BIN]=0.,0.
			Mcorr[BIN]=1.
		else:	
			SHEAR[BIN]= np.sum(shear*np.cos(2.*angle[maskr])*w2)/np.sum((np.cos(2.*angle[maskr])**2)*w2)
			CERO[BIN] = np.sum(cero*np.sin(2.*angle[maskr])*w2)/np.sum((np.sin(2.*angle[maskr])**2)*w2)
			sigma_e=(0.28**2.)
			ERR2=(ERR*sigma_e).sum()
			err[BIN]=((ERR2)/((pes2.sum())**2))**0.5			
			Mcorr[BIN]=1+np.average(mcorr,weights=w2)
			if booterror_flag:
				error_et[BIN],error_ex[BIN],etboot0,exboot0 = qbootstrap_errors(shear,cero,w2,angle[maskr],nboot)
				etboot += [etboot0]
				exboot += [exboot0]
			else:
				error_et[BIN],error_ex[BIN] = err[BIN],err[BIN]
		
		
	return [R,SHEAR/Mcorr,CERO/Mcorr,err/Mcorr,nbin,error_et/Mcorr,error_ex/Mcorr,N]	
