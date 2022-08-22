'''
Computes the volume integrated advective fluxes of salt squared, volume
mean salinity variance, volume averaged salinity variance, and cross-
advection for a 3D control volume over the TXLA shelf corresponding
to the location of the nested grid. 

See Theory Section of Schlichting et al. for the typesetted definitions.
Notes:
salt squared: \iint_{A_l} (us^2) \cdot dA
volume-mean salinity variance: \iint_{A_l} (us^{\prime^2}) \cdot dA
volume averaged salinity squared: \overline{s}^2 \iint_{A_l} (us) \cdot dA
cross-advection: \overline{s} \iint_{A_l} (us^\prime) \cdot dA
'''
#Packages 
import numpy as np
import xgcm
from xgcm import Grid
import xarray as xr
import xroms
from xhistogram.xarray import histogram
from glob import glob

#Open model output
path = glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver1/1hr/ocean_avg_0000*.nc')
# path = glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver1/30min/ocean_avg_0000*.nc')
# path = glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver1/10min/ocean_avg_0000*.nc')
ds = xroms.open_mfnetcdf(path)
ds, grid = xroms.roms_dataset(ds)

#Indexing for nested grid
xislice = slice(271,404)
etaslice = slice(31,149)

#Start of functions needed for advection
def volume_flux(ds, grid, xislice, etaslice):
    '''
Computes the volume advection through the lateral boundaries of a control volume for ROMS model output using
average files. We use the volume-conserving fluxes Huon and Hvom to increase the numerical accuracy. 
-----
Input: 
ds - xarray roms dataset
grid - xgcm grid of roms output
xislice - slice object of desired xi points
etaslice -slice object of desired eta points
-----
Output:
Qds: Xarray dataset of volume flux at the four horizontal control surfaces
vol_adv: Xarray dataarray of the net volume flux as a function of time only
    '''
    uflux = ds.Huon.isel(eta_rho = etaslice, xi_u = slice(xislice.start-1, xislice.stop))
    vflux = ds.Hvom.isel(eta_v = slice(etaslice.start-1, etaslice.stop), xi_rho = xislice)

    QW = uflux.isel(xi_u = 0) #West
    QE = uflux.isel(xi_u = -1) #East
    QN = vflux.isel(eta_v = -1) #North
    QS = vflux.isel(eta_v = 0) #South

    #Name individual components in case histograms are made, which require the variables be named
    QW.name = 'QW'
    QE.name = 'QE'
    QN.name = 'QN'
    QS.name = 'QS'

    Qds = xr.merge([QW, QE, QN, QS], compat='override')
    
    vol_adv = -(Qds.QW.sum(['eta_rho', 's_rho'])-Qds.QE.sum(['eta_rho', 's_rho']) \
                 +Qds.QS.sum(['xi_rho', 's_rho'])-Qds.QN.sum(['xi_rho', 's_rho']))
        
    return Qds, vol_adv

def salt_cv(ds, grid, xislice, etaslice):
    '''
Computes the boundary salinity of a control volume for ROMS model output. 
-----
Input: 
ds - xarray roms dataset
grid - xgcm grid of roms output
xislice - slice object of desired xi points
etaslice -slice object of desired eta points
-----
Output:
saltds: Xarray dataset of salinity at the four horizontal control surfaces. 

Notes: 
------
Interpolate the salt to the u and v points first, then slice the boundaries
    '''    
    su = grid.interp(ds.salt, 'X')
    sv = grid.interp(ds.salt, 'Y')

    su = su.isel(eta_rho = etaslice, xi_u = slice(xislice.start-1, xislice.stop)) 
    sv = sv.isel(eta_v = slice(etaslice.start-1, etaslice.stop), xi_rho = xislice)

    sW = su.isel(xi_u = 0) #West
    sE = su.isel(xi_u = -1) #East
    sN = sv.isel(eta_v = -1) #North
    sS = sv.isel(eta_v = 0) #South
   
    #DataArray Metadata
    sW.name = 'sW'
    sE.name = 'sE'
    sN.name = 'sN'
    sS.name = 'sS'
    
    saltds = xr.merge([sW, sE, sN, sS], compat='override')
    return saltds

def salt2_cv(ds, grid, xislice, etaslice):
    '''
Computes the boundary salinity squared of a control volume for ROMS model output. 
-----
Input: 
ds - xarray roms dataset
grid - xgcm grid of roms output
xislice - slice object of desired xi points
etaslice -slice object of desired eta points
-----
Output:
saltds: Xarray dataset of salinity at the four horizontal control surfaces. 

Notes:
-----
Square the salt, then interpolate. Could do the other way around and this will 
yield *slightly* different values, but not significant enough to change structure
    '''
    #Interpolate salt to the u and v points so that they can be multiplied by the volume fluxes
    su = grid.interp(ds.salt**2, 'X')
    sv = grid.interp(ds.salt**2, 'Y')

    su = su.isel(eta_rho = etaslice, xi_u = slice(xislice.start-1, xislice.stop)) 
    sv = sv.isel(eta_v = slice(etaslice.start-1, etaslice.stop), xi_rho = xislice)

    sW = su.isel(xi_u = 0) #West face of control volume
    sE = su.isel(xi_u = -1) #East face of control volume
    sN = sv.isel(eta_v = -1) #North face of control volume
    sS = sv.isel(eta_v = 0) #South face of control volume
   
    #DataArray Metadata
    sW.name = 'sW'
    sE.name = 'sE'
    sN.name = 'sN'
    sS.name = 'sS'
    
    salt2ds = xr.merge([sW, sE, sN, sS], compat='override')
    return salt2ds

def salt_flux(saltds, Qds, salt2ds):
    '''
Computes the boundary salinity transport of a control volume for ROMS model output. 
-----
Input: 
saltds - Xarray dataset of the salinity at the boundaries
Qds - Xarray dataset of the voume flux at the boundaries
-----
Output:
Qsds: Xarray dataset of salinity transport at the four horizontal control surfaces. 
Qssds: Xarray dataset of salinity squared transport at the four horizontal control surfaces. 
-----
    '''
    #Salt flux
    QsW = saltds.sW*Qds.QW
    QsE = saltds.sE*Qds.QE
    QsN = saltds.sN*Qds.QN
    QsS = saltds.sS*Qds.QS
    
    QsW.name = 'QsW'
    QsE.name = 'QsE'
    QsN.name = 'QsN'
    QsS.name = 'QsS'
    
    QssW = (salt2ds.sW)*Qds.QW
    QssE = (salt2ds.sE)*Qds.QE
    QssN = (salt2ds.sN)*Qds.QN
    QssS = (salt2ds.sS)*Qds.QS
    
    QssW.name = 'QssW'
    QssE.name = 'QssE'
    QssN.name = 'QssN'
    QssS.name = 'QssS'
    
    Qsds = xr.merge([QsW, QsE, QsN, QsS], compat='override')
    Qssds = xr.merge([QssW, QssE, QssN, QssS], compat='override')
    
    return Qsds, Qssds

def Qcsvar_faces(ds, xislice, etaslice, saltds, Qds):
    '''
Computes the boundary fluxes of salinity variance for a control volume of ROMS output. 
-----
Input: 
ds - xarray dataset
grid - xgcm grid
xislice - slice object of desired xi grid points
etaslice - slice object of desired eta grid points
saltds - salinity at each face of the control volume
Qds - volume flux at each face of the control volume
-----
Output:
Qsvards: salinity variance transport at each face of the control volume
-----
Notes: 
sprime^2 = (s-sbar)^2, where sbar is the volume averaged salinity.
Need to compute sbar for the control volume, then the variance fluxes at the boundaries
    '''
    #Compute volume-averaged salinity
    dV = (ds.dV).isel(eta_rho = etaslice, xi_rho = xislice)
 
    V = dV.sum(dim = ['eta_rho', 's_rho', 'xi_rho'])
    salt = ds.salt.isel(eta_rho = etaslice, xi_rho = xislice)
    sbar = (1/V)*(salt*dV).sum(dim = ['eta_rho', 'xi_rho','s_rho'])

    #Use the salt at the bondaries to compute the variance
    svarW = ((saltds.sW-sbar)**2)
    svarE = ((saltds.sE-sbar)**2)
    svarN = ((saltds.sN-sbar)**2)
    svarS = ((saltds.sS-sbar)**2)
    
    #Multiply the variance by the volume flux to get the variance boundary flux
    QsvarW = Qds.QW*svarW
    QsvarE = Qds.QE*svarE
    QsvarN = Qds.QN*svarN
    QsvarS = Qds.QS*svarS
    
    QsvarW.name = 'QsvarW'
    QsvarE.name = 'QsvarE'
    QsvarN.name = 'QsvarN'
    QsvarS.name = 'QsvarS'

    Qsvards = xr.merge([QsvarW, QsvarE, QsvarN, QsvarS], compat='override')
    
    return Qsvards

def Qsbarsprime2_advection(ds, xislice, etaslice, saltds, Qds):
    '''
Computes the boundary fluxes of 2*sbar*sprime for a control volume of ROMS output. 
-----
Input: 
ds - xarray dataset
grid - xgcm grid
xislice - slice object of desired xi grid points
etaslice - slice object of desired eta grid points
saltds - salinity at each face of the control volume
Qds - volume flux at each face of the control volume
-----
Output:
sbarsprime_adv
-----
Notes: 
sbarsprime_adv = -2*sbar*sprime, where sbar is the volume averaged salinity.
    '''
    #sbar first
    dV = ds.dV.isel(eta_rho = etaslice, xi_rho = xislice)
    V = dV.sum(dim = ['eta_rho', 's_rho', 'xi_rho'])
    salt = ds.salt.isel(eta_rho = etaslice, xi_rho = xislice)
    sbar = (1/V)*(salt*dV).sum(dim = ['eta_rho', 'xi_rho','s_rho'])

    #Use the salt at the bondaries to compute the variance
    sprimeW = (saltds.sW-sbar)
    sprimeE = (saltds.sE-sbar)
    sprimeN = (saltds.sN-sbar)
    sprimeS = (saltds.sS-sbar)
    
    #Multiply the variance by the volume flux to get the variance boundary flux
    QsbarprimeW = Qds.QW*(sprimeW*sbar)
    QsbarprimeE = Qds.QE*(sprimeE*sbar)
    QsbarprimeN = Qds.QN*(sprimeN*sbar)
    QsbarprimeS = Qds.QS*(sprimeS*sbar)
    
    QsbarprimeW.name = 'QsbarprimeW'
    QsbarprimeE.name = 'QsbarprimeE'
    QsbarprimeN.name = 'QsbarprimeN'
    QsbarprimeS.name = 'QsbarprimeS'

    Qsbarprimeds = xr.merge([QsbarprimeW, QsbarprimeE, QsbarprimeN, QsbarprimeS], compat='override')
    
    #Distribute the two after integrating - could do it before and will yield slightly
    #different values
    sbarprime_adv = -2*(Qsbarprimeds.QsbarprimeW.sum(['eta_rho', 's_rho'])-Qsbarprimeds.QsbarprimeE.sum(['eta_rho', 's_rho']) \
                   +Qsbarprimeds.QsbarprimeS.sum(['xi_rho', 's_rho'])-Qsbarprimeds.QsbarprimeN.sum(['xi_rho', 's_rho']))
    return sbarprime_adv

#-----------------------------
#Run the functions defined above! Then remove all attributes, which eliminates the grid metrics
#for saving to a .nc file. If you don't do that, it will raise an error when saving. 

# Salinity for the four control surfaces 
saltds = salt_cv(ds, grid, xislice, etaslice)

# Salinity squared for the four control surfaces 
salt2ds = salt2_cv(ds, grid, xislice, etaslice)

# Volume fluxes for the four control surfaces 
Qds, vol_adv = volume_flux(ds, grid, xislice, etaslice) 

# Volume advection that is volume integrated 
voladv_xr = -(Qds.QW.sum(['eta_rho', 's_rho'])-Qds.QE.sum(['eta_rho', 's_rho']) \
             +Qds.QS.sum(['xi_rho', 's_rho'])-Qds.QN.sum(['xi_rho', 's_rho']))
voladv_xr.attrs = ''

# Salt advection that is volume integrated
Qsds, Qssds = salt_flux(saltds, Qds, salt2ds)
sadv_xr = -(Qsds.QsW.sum(['eta_rho', 's_rho'])-Qsds.QsE.sum(['eta_rho', 's_rho']) \
             +Qsds.QsS.sum(['xi_rho', 's_rho'])-Qsds.QsN.sum(['xi_rho', 's_rho']))
sadv_xr.attrs = ''

# Salt squared advection that is volume integrated
ssadv_xr = -(Qssds.QssW.sum(['eta_rho', 's_rho'])-Qssds.QssE.sum(['eta_rho', 's_rho']) \
             +Qssds.QssS.sum(['xi_rho', 's_rho'])-Qssds.QssN.sum(['xi_rho', 's_rho']))
ssadv_xr.attrs = ''

# Volume-mean salinity variance advection that is volume integrated
Qsvards = Qcsvar_faces(ds, xislice, etaslice, saltds, Qds)
svaradv_xr = -(Qsvards.QsvarW.sum(['eta_rho', 's_rho'])-Qsvards.QsvarE.sum(['eta_rho', 's_rho']) \
             +Qsvards.QsvarS.sum(['xi_rho', 's_rho'])-Qsvards.QsvarN.sum(['xi_rho', 's_rho']))
svaradv_xr.attrs = ''

# Open the volume-averaged salinity to compute the sbar^2 advection by simplying squaring and
# multiplying by the volume-advection
sbar = xr.open_mfdataset('/d2/home/dylan/JAMES/budget_outputs/sbar/sbar_parent_ver1_2010_*.nc').sbar
# sbar = xr.open_mfdataset('/d2/home/dylan/JAMES/budget_outputs/30min/sbar/sbar_parent_2010_30min_*.nc').sbar
# sbar = xr.open_mfdataset('/d2/home/dylan/JAMES/budget_outputs/10min/sbar/sbar_parent_2010_10min_*.nc').sbar
sbar2_adv =  ((sbar**2)*vol_adv)
sbar2_adv.attrs = ''

# 2 sbar sprime advection
sbarprime_adv = Qsbarsprime2_advection(ds, xislice, etaslice, saltds, Qds)
sbarprime_adv.attrs = ''

#Subset the data and save to a netcdf every day. Note that this might have to be down or -up 
#sampled depending on the laptop or cluster you use. 
print('saving outputs')
dates = np.arange('2010-06-03', '2010-07-15', dtype = 'datetime64[D]') 
for d in range(len(dates)):    
    #Salt squared
    ssadv_xr_sel = ssadv_xr.sel(ocean_time = str(dates[d]))
    path = '/d2/home/dylan/JAMES/budget_outputs/advection/saltsquareadv_parent_ver1_2010_%s.nc' %d
#     path = '/d2/home/dylan/JAMES/budget_outputs/30min/advection/saltsquareadv_parent_2010_30min_%s.nc' %d
    # path = '/d2/home/dylan/JAMES/budget_outputs/10min/advection/saltsquareadv_parent_2010_10min_%s.nc' %d
    ssadv_xr_sel.to_netcdf(path, mode = 'w')

    #Salt anomaly squared
    svaradv_xr_sel = svaradv_xr.sel(ocean_time = str(dates[d]))
    path = '/d2/home/dylan/JAMES/budget_outputs/advection/saltvaradv_parent_ver1_2010_%s.nc' %d
#     path = '/d2/home/dylan/JAMES/budget_outputs/30min/advection/saltvaradv_parent_2010_30min_%s.nc' %d
    # path = '/d2/home/dylan/JAMES/budget_outputs/10min/advection/saltvaradv_parent_2010_10min_%s.nc' %d
    svaradv_xr_sel.to_netcdf(path, mode = 'w')
    
    #Extra terms 
    sbar2_adv_sel = sbar2_adv.sel(ocean_time = str(dates[d]))
    path = '/d2/home/dylan/JAMES/budget_outputs/advection/sbar2_advection_ver1_2010_%s.nc' %d
#     path = '/d2/home/dylan/JAMES/budget_outputs/30min/advection/sbar2_advection_2010_30min_%s.nc' %d
    # path = '/d2/home/dylan/JAMES/budget_outputs/10min/advection/sbar2_advection_2010_10min_%s.nc' %d
    sbar2_adv_sel.name = 'sbar2_advection'
    sbar2_adv_sel.to_netcdf(path, mode = 'w')

    sbarprime_adv_sel = sbarprime_adv.sel(ocean_time = str(dates[d]))
    path = '/d2/home/dylan/JAMES/budget_outputs/advection/2sbarsprime_advection_ver1_2010_%s.nc' %d
#     path = '/d2/home/dylan/JAMES/budget_outputs/30min/advection/2sbarsprime_advection_2010_30min_%s.nc' %d
    # path = '/d2/home/dylan/JAMES/budget_outputs/10min/advection/2sbarsprime_advection_2010_10min_%s.nc' %d
    sbarprime_adv_sel.name = '2sbarprime_advection'
    sbarprime_adv_sel.to_netcdf(path, mode = 'w')
    