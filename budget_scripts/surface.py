'''
Computes the surface salt squared, volume-mean salt variance,
and extra surface fluxes for the TXLA model. See 'Theory' of
Schlichting et al. JAMES for the equations'.

Notes:
-----
Fluxes out of the control volume are considered positive.
evaporation: 'positivbe value: upward flux, salting (evaporation)', 'negative_value :downward flux, freshening (condensation)'
rain: 'positive_value: downward flux, freshening (precipitation)'
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
path = glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver0/ocean_avg_0000*.nc')
ds_avg = xroms.open_mfnetcdf(path)
ds_avg, grid = xroms.roms_dataset(ds_avg)

#Slices
xislice = slice(271,404)
etaslice = slice(31,149)

#Salt squared surface flux: (2*s*s*(E-P)). First need to extract E-P and divide by the
# density of freshwater, which is assumed to be 1000 kg/m^3.
dA = ds_avg.dA.isel(eta_rho = etaslice, xi_rho = xislice)
eminusp_avg = (ds_avg.evaporation-ds_avg.rain)/1000
s2flux_avg = eminusp_avg*ds_avg.salt.isel(s_rho = -1)*2*ds_avg.salt.isel(s_rho = -1)
s2flux_int_avg = (s2flux_avg*dA).isel(ocean_time = slice(1,-1)).sum(['eta_rho', 'xi_rho'])
s2flux_int_avg.attrs = ''

#S^prime^2 surface flux: (2*s^prime*s*(E-P))
dV = ds_avg.dV.isel(eta_rho = etaslice, xi_rho = xislice)
V = dV.sum(dim = ['eta_rho', 's_rho', 'xi_rho'])
salt = ds_avg.salt.isel(eta_rho = etaslice, xi_rho = xislice)
sbar = (1/V)*(salt*dV).sum(dim = ['eta_rho', 'xi_rho','s_rho'])
sprime = ds_avg.salt.isel(eta_rho = etaslice, xi_rho = xislice, s_rho = -1)-sbar
sprime2flux_avg = 2*(salt.isel(s_rho = -1))*sprime*eminusp_avg
sprime2flux_int_avg = (sprime2flux_avg*dA).isel(ocean_time = slice(1,-1)).sum(['eta_rho', 'xi_rho'])
sprime2flux_int_avg.attrs = ''

#Extra terms:
surf_extra = ((2*sbar*sprime)+(2*sbar**2))
surf_extra_int = (surf_extra*dA*eminusp_avg).isel(ocean_time = slice(1,-1)).sum(['eta_rho', 'xi_rho'])
surf_extra_int.attrs = ''

#Subset the data daily and save to a netcdf file. The reason we do this is to avoid overloading
#the cluseter.
print('saving outputs')
dates = np.arange('2010-06-03', '2010-07-15', dtype = 'datetime64[D]')
for d in range(len(dates)):
    #s^2
    s2flux_int_avg_sel = s2flux_int_avg.sel(ocean_time = str(dates[d]))
    s2flux_int_avg_sel.name = 's2flux'
    path = '/d2/home/dylan/JAMES/budget_outputs/surface_fluxes/s2flux_2010_%s.nc' %d
    s2flux_int_avg_sel.to_netcdf(path, mode = 'w')

    #s^prime^2
    sprime2flux_int_avg_sel = sprime2flux_int_avg.sel(ocean_time = str(dates[d]))
    sprime2flux_int_avg_sel.name = 'sprime2flux'
    path = '/d2/home/dylan/JAMES/budget_outputs/surface_fluxes/sprime2flux_2010_%s.nc' %d
    sprime2flux_int_avg_sel.to_netcdf(path, mode = 'w')

    #Extra terms
    surf_extra_sel = surf_extra_int.sel(ocean_time = str(dates[d]))
    path = '/d2/home/dylan/JAMES/budget_outputs/surface_fluxes/extra_2010_%s.nc' %d
    surf_extra_sel.name = 'surface_fluxes_extra'
    surf_extra_sel.to_netcdf(path, mode = 'w')
