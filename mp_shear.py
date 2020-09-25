"""
New multipole share parallel calculation
"""

import numpy as np

from astropy.cosmology import LambdaCDM
# from scipy.misc import derivative
# from scipy import integrate
import profiles_fit

class mp_shear:
    """
    Equations from van Uitert (vU, arXiv:1610.04226) for the
    multipole expansion and from Ford et al. (F,2015) for
    the misscentring

    INPUTS:
    r               [Mpc] - Float or numpy_array
                    Distance to the centre of
                    the gravitational potential

    OPTIONAL:
    M200            [M_sun] - Float/ Cluster Mass
    ellip           Float / Halo ellipticity defined as
                    (1-q)/(1+q) where q is the semi-axis
                    ratio (q < 1)
    z               Float/ Lens redshift
    zs              Float/ Source redshift
    h               Float/ Cosmological quantity
    misscentred     Float/ If True computes de misscentred quantities
                    The misscentred is considered only in the
                    x - axis
    s_off           [Mpc h-1] Float/ sigma_offset width of the distribution
                    of cluster offsets (F_Eq11)
    components      List of misscentred components that are going
                    to be computed:
                    't0'   -> Gt0_off
                    't'    -> Gt_off
                    'tcos' -> Gt_off_cos
                    'xsin' -> Gx_off_sin
    verbose         Print time computing
    Yanmiss         if True use Eq 4 to model the miss of Yan et al. 2020

    OUTPUT:
    output          Dictionary (the shear is scales with the
                    critical density Gamma_i = Sigma_cr*gamma_i
                    ---------- vU_Eq18 ---------
                    Gt0: Shear monopole
                    Gt2: Tangential component for the quadrupole
                    Gx2: Cross component for the quadrupole
                    --------- Misscentred terms --------
                    Gt0_off: (ellip = 0) F_Eq14
                    Gt_off: Integrated tangential component (vU_Eq17)
                    for considering kappa_offset
                    Gt_off_cos: Integrated tangential component times
                    cos(2theta) (vU_Eq17) for considering kappa_offset
                    Gx_off_sin: Integrated cross component times
                    sin(2theta) (vU_Eq17) for considering kappa_offset
    """

    cvel = 299792458    # Speed of light (m.s-1)
    G    = 6.670e-11    # Gravitational constant (m3.kg-1.s-2)
    pc   = 3.085678e16  # 1 pc (m)
    Msun = 1.989e30     # Solar mass (kg)

    def __init__(self, r, M200=1.e14, ellip=0.25, z=0.2, h=0.7,
                        misscentred=False, s_off=0.4,
                        components = ['t0', 't', 'tcos', 'xsin'],
                        verbose=True, Yanmiss = False):
        """
        init method, it sets up all variables
        """

        if not isinstance(r, (np.ndarray)):
            r = np.array([r])

        self.r           = r
        self.M200        = M200
        self.ellip       = ellip
        self.z           = z
        self.h           = h
        self.misscentred = misscentred
        self.s_off       = s_off
        self.components  = components
        self.verbose     = verbose
        self.Yanmiss     = Yanmiss

        # Compute cosmological parameters
        self.cosmo    = LambdaCDM(H0=self.h*100, Om0=0.3, Ode0=0.7)
        self.H        = self.cosmo.H(self.z).value / (1.0e3 * self.pc) #H at z_pair s-1
        self.roc      = (3.0 * (self.H**2.0)) / (8.0*np.pi * self.G) #critical density at z_pair (kg.m-3)
        self.roc_mpc  = self.roc * ((self.pc * 1.0e6)**3.0)

        # Compute R_200
        self.R200 = profiles_fit.r200_nfw(self.M200, self.roc_mpc)

        # Scaling sigma_off
        self.s_off = self.s_off / self.h

    #
    # def Delta_Sigma(R):
    #     '''
    #     Density contraste for NFW
    #     '''
    #
    #     #calculo de c usando la relacion de Duffy et al 2008
    #
    #     M = ((800.0 * np.pi * roc_mpc * (R200**3)) / (3.0 * self.Msun)) * self.h
    #     c = 5.71*((M/2.e12)**-0.084)*((1.+z)**-0.47)
    #
    #     ####################################################
    #
    #     deltac = (200.0 / 3.0) * ( (c**3.0) / ( np.log(1.0 + c) - (c / (1.0 + c))))
    #     x = np.round((R * c) / R200, 12)
    #     m1 = x < 1.0
    #     m2 = x > 1.0
    #     m3 = x == 1.0
    #
    #     try:
    #         jota=np.zeros(len(x))
    #         atanh=np.arctanh(((1.0-x[m1])/(1.0+x[m1]))**0.5)
    #         jota[m1]=(4.0*atanh)/((x[m1]**2.0)*((1.0-x[m1]**2.0)**0.5)) \
    #             + (2.0*np.log(x[m1]/2.0))/(x[m1]**2.0) - 1.0/(x[m1]**2.0-1.0) \
    #             + (2.0*atanh)/((x[m1]**2.0-1.0)*((1.0-x[m1]**2.0)**0.5))
    #         atan=np.arctan(((x[m2]-1.0)/(1.0+x[m2]))**0.5)
    #         jota[m2]=(4.0*atan)/((x[m2]**2.0)*((x[m2]**2.0-1.0)**0.5)) \
    #             + (2.0*np.log(x[m2]/2.0))/(x[m2]**2.0) - 1.0/(x[m2]**2.0-1.0) \
    #             + (2.0*atan)/((x[m2]**2.0-1.0)**1.5)
    #         jota[m3]=2.0*np.log(0.5)+5.0/3.0
    #     except:
    #         if m1:
    #             atanh=np.arctanh(((1.0-x[m1])/(1.0+x[m1]))**0.5)
    #             jota = ((4.0*atanh)/((x[m1]**2.0)*((1.0-x[m1]**2.0)**0.5))
    #                 + (2.0*np.log(x[m1]/2.0))/(x[m1]**2.0) - 1.0/(x[m1]**2.0-1.0)
    #                 + (2.0*atanh)/((x[m1]**2.0-1.0)*((1.0-x[m1]**2.0)**0.5)))
    #         if m2:
    #             atan=np.arctan(((x[m2]-1.0)/(1.0+x[m2]))**0.5)
    #             jota = ((4.0*atan)/((x[m2]**2.0)*((x[m2]**2.0-1.0)**0.5))
    #                 + (2.0*np.log(x[m2]/2.0))/(x[m2]**2.0) - 1.0/(x[m2]**2.0-1.0)
    #                 + (2.0*atan)/((x[m2]**2.0-1.0)**1.5))
    #         if m3:
    #             jota = 2.0*np.log(0.5)+5.0/3.0
    #
    #
    #
    #     rs_m=(R200*1.e6*self.pc)/c
    #     kapak=((2.*rs_m*deltac*roc_mpc)*(self.pc**2/Msun))/((self.pc*1.0e6)**3.0)
    #     return kapak*jota
    #
    #
    # def monopole(R):
    #     '''
    #     Projected density for NFW
    #
    #     '''
    #     if not isinstance(R, (np.ndarray)):
    #         R = np.array([R])
    #
    #     # m = R == 0.
    #     # R[m] = 1.e-8
    #
    #     #calculo de c usando la relacion de Duffy et al 2008
    #
    #     M=((800.0*np.pi*roc_mpc*(R200**3))/(3.0*Msun))*h
    #     c=5.71*((M/2.e12)**-0.084)*((1.+z)**-0.47)
    #
    #     ####################################################
    #
    #     deltac=(200./3.)*( (c**3) / ( np.log(1.+c)- (c/(1+c)) ))
    #
    #     x=(R*c)/R200
    #     m1 = x <= (1.0-1.e-12)
    #     m2 = x >= (1.0+1.e-12)
    #     m3 = (x == 1.0)
    #     m4 = (~m1)*(~m2)*(~m3)
    #
    #     jota  = np.zeros(len(x))
    #     atanh = np.arctanh(np.sqrt((1.0-x[m1])/(1.0+x[m1])))
    #     jota[m1] = (1./(x[m1]**2-1.))*(1.-(2./np.sqrt(1.-x[m1]**2))*atanh)
    #
    #     atan = np.arctan(((x[m2]-1.0)/(1.0+x[m2]))**0.5)
    #     jota[m2] = (1./(x[m2]**2-1.))*(1.-(2./np.sqrt(x[m2]**2 - 1.))*atan)
    #
    #     jota[m3] = 1./3.
    #
    #     x1 = 1.-1.e-4
    #     atanh1 = np.arctanh(np.sqrt((1.0-x1)/(1.0+x1)))
    #     j1 = (1./(x1**2-1.))*(1.-(2./np.sqrt(1.-x1**2))*atanh1)
    #
    #     x2 = 1.+1.e-4
    #     atan2 = np.arctan(((x2-1.0)/(1.0+x2))**0.5)
    #     j2 = (1./(x2**2-1.))*(1.-(2./np.sqrt(x2**2 - 1.))*atan2)
    #
    #     jota[m4] = np.interp(x[m4].astype(float64),[x1,x2],[j1,j2])
    #
    #     rs_m=(R200*1.e6*self.pc)/c
    #     kapak=((2.*rs_m*deltac*roc_mpc)*(self.pc**2/Msun))/((self.pc*1.0e6)**3.0)
    #     return kapak*jota
    #
    # def quadrupole(R):
    #     '''
    #     Quadrupole term defined as (d(Sigma)/dr)*r
    #
    #     '''
    #     m0p = derivative(monopole,R,dx=1e-5)
    #     return m0p*R
