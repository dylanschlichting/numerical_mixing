'''
This notebook computes tendency terms for the salt squared and salt anomaly squared budgets 
using purely average files, NOT a mix of diagnostic and average files. Saves d/dt (s^2,), d/dt (s^'^2')
d/dt (2sbars'), and d/dt (sbar**2) that are all volume integrated.

The code is structured such that the individual tracers, i.e. s^2 or sprime^2, are saved 
and repoened as netcdf files because we have to calculate the time derivatives as finite differences,
so we need to the first two and last two values for the computations. 

Notes
-----
1) We need to be very careful about the times used for the derivatives. All other tracer budget terms are computed
from June 3, 00:30 to July 13, 23:30. Make sure that all tracers are sliced in time to accomadate this after 
you open them.
2) Uncomment any for loops to rerun the tracer calculations, I turned them off for messing with time-indexing. 
3) timestep for finite differences = 2*DT = 2*3600 s / hr. 
4) Variables relevant to the calculations are saved individually because they are loaded into memory for the 
calculation of the time derivatives using numpy. You could rewrite this using purely xarray to shorten this script
considerably. 
5) All time derivatives are computed as rate of change of the total tracer content in a cell, i.e., (d(c delta)/dt)/delta,
where delta is the vertical layer thickness (dz).
'''

#Packages 
import numpy as np
import xgcm
from xgcm import Grid
import xarray as xr
import xroms
from glob import glob

#Open model output
path = glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver1/1hr/ocean_avg_0000*.nc')
ds_avg = xroms.open_mfnetcdf(path)
ds_avg, grid = xroms.roms_dataset(ds_avg)

#Slices in the xi and eta directions corresponding to the location of the nested grid 
xislice = slice(271,404)
etaslice = slice(31,149)

#Save delta
delta = ds_avg.dz.isel(eta_rho = etaslice, xi_rho = xislice)
delta.attrs = ''
dates = np.arange('2010-06-03', '2010-07-14', dtype = 'datetime64[D]') 
for d in range(len(dates)):    
    delta_sel = delta.sel(ocean_time = str(dates[d]))
    delta_sel.name = 'dV'
    path = '/d2/home/dylan/JAMES/budget_outputs/tracers/delta_ver1_2010_%s.nc' %d
    delta_sel.to_netcdf(path, mode = 'w')

delta = xr.open_mfdataset('/d2/home/dylan/JAMES/budget_outputs/tracers/delta_ver1_2010*.nc').dV.sel(ocean_time = slice('2010-06-03', '2010-07-13'))

#Save dV
dV = ds_avg.dV.isel(eta_rho = etaslice, xi_rho = xislice)
dV.attrs = ''
dates = np.arange('2010-06-03', '2010-07-14', dtype = 'datetime64[D]') 
for d in range(len(dates)):    
    dV_sel = dV.sel(ocean_time = str(dates[d]))
    dV_sel.name = 'dV'
    path = '/d2/home/dylan/JAMES/budget_outputs/tracers/dV_ver1_2010_%s.nc' %d
    dV_sel.to_netcdf(path, mode = 'w')

dV = xr.open_mfdataset('/d2/home/dylan/JAMES/budget_outputs/tracers/dV_ver1_2010*.nc').dV.sel(ocean_time = slice('2010-06-03', '2010-07-13'))
V = dV.sum(dim = ['eta_rho', 's_rho', 'xi_rho'])

#Save salt squared
s2 = ((ds_avg.salt.isel(eta_rho = etaslice, xi_rho = xislice))**2)
s2.attrs = ''
print('saving outputs')
for d in range(len(dates)):    
    s2_sel = s2.sel(ocean_time = str(dates[d]))
    s2_sel.name = 's2'
    path = '/d2/home/dylan/JAMES/budget_outputs/tracers/s2_ver1_2010_%s.nc' %d
    s2_sel.to_netcdf(path, mode = 'w')

#Compute ds^2/dt
s2 = xr.open_mfdataset('/d2/home/dylan/JAMES/budget_outputs/tracers/s2_ver1_2010_*.nc').s2.sel(ocean_time = slice('2010-06-03', '2010-07-13'))
s2v = s2*delta
ds2vdt = (((s2v).values[2:] - (s2v).values[:-2])/(2*3600))
ds2vdt_v =  ds2vdt/(delta.values[1:-1])
ds2vdt_v_int = (ds2vdt_v*dV[1:-1]).sum(axis = (1,2,3))

np.save('/d2/home/dylan/JAMES/budget_outputs/tendency/ds2dt_parent_ver1_2010.npy', ds2vdt_v_int)

sbar = xr.open_mfdataset('/d2/home/dylan/JAMES/budget_outputs/sbar/sbar_parent_ver1_2010_*.nc').sbar.sel(ocean_time = slice('2010-06-03', '2010-07-13'))

#Compute sprime2 and save
salt = ds_avg.salt.isel(eta_rho = etaslice, xi_rho = xislice)
sprime = ds_avg.salt-sbar
sprime2 = (sprime**2).isel(eta_rho = etaslice, xi_rho = xislice)
sprime2.attrs = ''

for d in range(len(dates)):    
    sprime2_sel = sprime2.sel(ocean_time = str(dates[d]))
    sprime2_sel.name = 'sprime2'
    path = '/d2/home/dylan/JAMES/budget_outputs/tracers/sprime2_ver1_2010_%s.nc' %d
    sprime2_sel.to_netcdf(path, mode = 'w')

#Move on to dsprime2dt
sprime2 = xr.open_mfdataset('/d2/home/dylan/JAMES/budget_outputs/tracers/sprime2_ver1_2010_*.nc').sprime2.sel(ocean_time = slice('2010-06-03', '2010-07-13'))
sprime2v = sprime2*delta
dsprime2vdt = (((sprime2v).values[2:] - (sprime2v).values[:-2])/(2*3600))
dsprime2vdt_v =  dsprime2vdt/(delta.values[1:-1])
dsprime2vdt_v_int = (dsprime2vdt_v*dV[1:-1]).sum(axis = (1,2,3))

np.save('/d2/home/dylan/JAMES/budget_outputs/tendency/dsprime2dt_ver1_parent_2010.npy', dsprime2vdt_v_int)

#Compute the extra terms. Start with d(sbar^2)/dt V. We could compute this without the delta portion and pull outside the 
#volume integral since sbar has no spatial gradients. Doesn't impact the answer so left in this form.
sbar2v = sbar**2*delta
dsbar2vdt = (((sbar2v).values[2:] - (sbar2v).values[:-2])/(2*3600))
dsbar2vdt_v =  dsbar2vdt/(delta.values[1:-1])
dsbar2vdt_v_int = (dsbar2vdt_v*dV[1:-1]).sum(axis = (1,2,3))
np.save('/d2/home/dylan/JAMES/budget_outputs/tendency/dsbar2dt_parent_ver1_2010.npy', dsbar2vdt_v_int)

#Cross tendency
sprime = xr.open_mfdataset('/d2/home/dylan/JAMES/budget_outputs/tracers/sprime_ver1_2010_*.nc').sprime.sel(ocean_time = slice('2010-06-03', '2010-07-13'))
sprimev = sprime*delta
dsprimevdt = (((sprimev).values[2:] - (sprimev).values[:-2])/(2*3600))
dsprimevdt_v =  dsprimevdt/(delta.values[1:-1])
dsprimevdt_v_int = (2*sbar[1:-1,np.newaxis,np.newaxis,np.newaxis]*dsprimevdt_v*dV[1:-1]).sum(axis = (1,2,3))
np.save('/d2/home/dylan/JAMES/budget_outputs/tendency/dsbarsprimedt1_parent_ver1_2010.npy', dsprimevdt_v_int)

