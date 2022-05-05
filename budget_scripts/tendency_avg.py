'''
This notebook computes tendency terms for the salt squared and salt anomaly squared budgets 
using purely average files, NOT a mix of diagnostic and average files. Saves d/dt (s^2,), d/dt (s^'^2')
d/dt (2sbars'), and d/dt (sbar**2) that are all volume integrated.

The code is structured such that the individual tracers, i.e. s^2 or sprime^2, are saved 
and repoened as netcdf files because we have to calculate the time derivatives as finite differences,
so we need to the first two and last two values for the computations. 

*** ADDITIONAL NOTES ***
1) We need to be very careful about the times used for the derivatives. All other tracer budget terms are computed
from June 3, 00:30 to July 13, 23:30. Make sure that all tracers are sliced in time to accomadate this after 
you open them.
2) Uncomment any for loops to rerun the tracer calculations, I turned them off for messing with time-indexing. 
3) timestep for finite differences = 2*dt = 2*3600 s / hr. 
'''

#Packages 
import numpy as np
import xgcm
from xgcm import Grid
import xarray as xr
import xroms
from glob import glob

#Open model output
path = glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver0/ocean_avg_0000*.nc')
ds_avg = xroms.open_mfnetcdf(path)
ds_avg, grid = xroms.roms_dataset(ds_avg)

#Slices in the xi and eta directions corresponding to the location of the nested grid 
xislice = slice(271,404)
etaslice = slice(31,149)

#Save dV because it's needed for the volume integration
dV = ds_avg.dV.isel(eta_rho = etaslice, xi_rho = xislice)
dV.attrs = ''
dates = np.arange('2010-06-03', '2010-07-15', dtype = 'datetime64[D]') 
# for d in range(len(dates)):    
#     dV_sel = dV.sel(ocean_time = str(dates[d]))
#     dV_sel.name = 'dV'
#     path = '/d2/home/dylan/JAMES/budget_outputs/tracers/dV_2010_%s.nc' %d
#     dV_sel.to_netcdf(path, mode = 'w')

#Open the tracer and slice as discussed above.
dV = xr.open_mfdataset('/d2/home/dylan/JAMES/budget_outputs/tracers/dV_2010_*.nc').dV.sel(ocean_time = slice('2010-06-03', '2010-07-13'))
V = dV.sum(dim = ['eta_rho', 's_rho', 'xi_rho'])

#Move on to s^2
# s2 = ((ds_avg.salt.isel(eta_rho = etaslice, xi_rho = xislice))**2)
# s2.attrs = ''
# print('saving outputs')
# for d in range(len(dates)):    
#     s2_sel = s2.sel(ocean_time = str(dates[d]))
#     s2_sel.name = 's2'
#     path = '/d2/home/dylan/JAMES/budget_outputs/tracers/s2_2010_%s.nc' %d
#     s2_sel.to_netcdf(path, mode = 'w')

#ds^2/dt
s2 = xr.open_mfdataset('/d2/home/dylan/JAMES/budget_outputs/tracers/s2_2010_*.nc').s2.sel(ocean_time = slice('2010-06-03', '2010-07-13'))
ds2dt_alt = ((((s2).values[2:] - (s2).values[:-2])/(2*3600))*dV.values[1:-1]).sum(axis = (1,2,3))
np.save('/d2/home/dylan/JAMES/budget_outputs/ds2dt_parent_2010.npy', ds2dt_alt)

#Move on to dsprime2dt
sbar = xr.open_mfdataset('/d2/home/dylan/JAMES/budget_outputs/sbar/sbar_parent_2010_*.nc').sbar.sel(ocean_time = slice('2010-06-03', '2010-07-13'))
salt = ds_avg.salt.isel(eta_rho = etaslice, xi_rho = xislice)
sprime = ds_avg.salt-sbar
sprime2 = (sprime**2).isel(eta_rho = etaslice, xi_rho = xislice)
sprime2.attrs = ''

# print('saving outputs')
# for d in range(len(dates)):    
#     sprime2_sel = sprime2.sel(ocean_time = str(dates[d]))
#     sprime2_sel.name = 'sprime2'
#     path = '/d2/home/dylan/JAMES/budget_outputs/tracers/sprime2_2010_%s.nc' %d
#     sprime2_sel.to_netcdf(path, mode = 'w')

sprime2 = xr.open_mfdataset('/d2/home/dylan/JAMES/budget_outputs/tracers/sprime2_2010_*.nc').sprime2.sel(ocean_time = slice('2010-06-03', '2010-07-13'))
dsprime2dt_alt = ((((sprime2).values[2:] - (sprime2).values[:-2])/(2*3600))*dV.values[1:-1]).sum(axis = (1,2,3))
np.save('/d2/home/dylan/JAMES/budget_outputs/dsprime2dt_parent_2010.npy', dsprime2dt_alt)

#Extra terms: dsbar^2/dt and 2*sbar*ds^'/dt
sbar2 = sbar**2
dsbar2dt = ((((sbar2).values[2:] - (sbar2).values[:-2])/(2*3600))*V.values[1:-1])
np.save('/d2/home/dylan/JAMES/budget_outputs/dsbar2dt_parent_2010.npy', dsbar2dt)

#Save s' just to avoid numerical truncation errors. 
sprime_slice = sprime.isel(eta_rho = etaslice, xi_rho = xislice)
sprime_slice.attrs = ''
# print('saving outputs')
# for d in range(len(dates)):    
#     sprime_slice_sel = sprime_slice.sel(ocean_time = str(dates[d]))
#     sprime_slice_sel.name = 'sprime'
#     path = '/d2/home/dylan/JAMES/budget_outputs/tracers/sprime_2010_%s.nc' %d
#     sprime_slice_sel.to_netcdf(path, mode = 'w')

sprime = xr.open_mfdataset('/d2/home/dylan/JAMES/budget_outputs/tracers/sprime_2010_*.nc').sprime.sel(ocean_time = slice('2010-06-03', '2010-07-13'))
dsprimedt = ((((sprime).values[2:] - (sprime).values[:-2])/(2*3600))*dV.values[1:-1]).sum(axis = (1,2,3))
dsbarsprimedt = (2*sbar.isel(ocean_time = slice(1,-1))*dsprimedt)
np.save('/d2/home/dylan/JAMES/budget_outputs/dsbarsprimedt_parent_2010.npy', dsbarsprimedt)