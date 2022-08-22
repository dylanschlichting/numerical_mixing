#Packages 
import numpy as np
import xgcm
from xgcm import Grid
import xarray as xr
import xroms
from datetime import datetime
import glob
import pandas as pd
from xhistogram.xarray import histogram

path = glob.glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver1/1hr/ocean_avg_0000*.nc')
ds_avg_parent = xroms.open_mfnetcdf(path)
ds_avg_parent, grid_parent = xroms.roms_dataset(ds_avg_parent)

path = glob.glob('/d1/shared/TXLA_ROMS/numerical_mixing/nest/ver1/ocean_avg_child_0000*.nc')
ds_avg_child = xroms.open_mfnetcdf(path)
ds_avg_child, grid_avg_child = xroms.roms_dataset(ds_avg_child)

#Corresponding location of the parent model subset. See 'check_grid.ipynb' for more information
xisliceparent = slice(271,404)
etasliceparent = slice(31,149)
xislicechild = slice(8,677-8)
etaslicechild = slice(8,602-8)

#Match the times of the parent and child models. Drop the first timestep after
#each restart, then match parent / child. 
timedrop = [np.datetime64('2010-06-18T18:30:00.000000000'), 
            np.datetime64('2010-06-19T18:30:00.000000000'), 
            np.datetime64('2010-07-09T18:30:00.000000000')]

ds_avg_child = ds_avg_child.where((ds_avg_child.ocean_time!= timedrop[0])
                                & (ds_avg_child.ocean_time!= timedrop[1])
                                & (ds_avg_child.ocean_time!= timedrop[2]),
                                   drop=True)

ds_avg_parent = ds_avg_parent.where(ds_avg_child.ocean_time==ds_avg_parent.ocean_time)

#Bin sizes for each variable: use 150 bins for each variable
rvortbins = np.linspace(-5,5,150)
divbins = np.linspace(-5,5,150)
strainbins = np.linspace(0,8,150)
sgradbins = np.linspace(0,0.002,150)

def surface_vorticity(ds, grid):
    '''
Calculates the surface vertical vorticity normalized by 
the Coriolis frequency. 
----
Inputs:
ds - Xarray Dataset
grid - XGCM grid object 
----
Outputs:
rvort_psi: Normalized vorticity on the psi points 

    '''
    u = ds.u.isel(s_rho=-1)
    v = ds.v.isel(s_rho=-1)

    dudy = grid.derivative(u, 'Y')
    dvdx = grid.derivative(v, 'X')
    f_psi = xroms.to_psi(ds.f, grid)

    rvort_psi = (dvdx-dudy)/f_psi
    
    return rvort_psi 

def surface_divergence(ds, grid):
    '''
Calculates the surface divergence normalized by the Coriolis frequency. 
----
Inputs:
ds - Xarray Dataset
grid - XGCM grid object 
----
Outputs:
divergence: Normalized divergence on the psi points for the surface rho layer

    '''
    u = ds.u.isel(s_rho=-1)
    v = ds.v.isel(s_rho=-1)

    dudx = grid.derivative(u, 'X', boundary = 'extend')
    dudx_psi = xroms.to_psi(dudx, grid)
    dvdy = grid.derivative(v, 'Y', boundary = 'extend')
    dvdy_psi = xroms.to_psi(dvdy, grid)
    
    f_psi = xroms.to_psi(ds.f, grid)
    divergence = (dudx_psi+dvdy_psi)/f_psi
    
    return divergence 

def surface_strain(ds, grid):
    '''
Calculates the surface strain normalized by the Coriolis frequency. 
----
Inputs:
ds - Xarray Dataset
grid - XGCM grid object 
----
Outputs:
strain: Normalized strain on the psi points 

    '''
    u = ds.u.isel(s_rho=-1)
    v = ds.v.isel(s_rho=-1)

    dudx = grid.derivative(u, 'X', boundary = 'extend')
    dudx_psi = xroms.to_psi(dudx, grid)
    dvdy = grid.derivative(v, 'Y', boundary = 'extend')
    dvdy_psi = xroms.to_psi(dvdy, grid)
    
    dudy = grid.derivative(u, 'Y')
    dvdx = grid.derivative(v, 'X')
    
    f_psi = xroms.to_psi(ds.f, grid)
    strain = (((dudx_psi-dvdy_psi)**2+(dvdx+dudy)**2)**(1/2))/f_psi
    
    return strain

def surface_saltgradmag(ds, grid):
    '''
Calculates the surface horizontal salinity gradient magnitude normalized by 
the Coriolis frequency. 
----
Inputs:
ds - Xarray Dataset
grid - XGCM grid object 
----
Outputs:
sgradmag: horizontal salinity gradient magnitude on the psi points

    '''
    s = ds.salt.isel(s_rho=-1)

    dsdx = grid.derivative(s, 'X', boundary = 'extend')
    dsdx_psi = xroms.to_psi(dsdx, grid)
    dsdy = grid.derivative(s, 'Y', boundary = 'extend')
    dsdy_psi = xroms.to_psi(dsdy, grid)
    
    sgradmag = (dsdx_psi**2+dsdy_psi**2)**(1/2)
    
    return sgradmag


#Compute the surface salinity gradient magnitude
sgradmag_surf_parent = surface_saltgradmag(ds_avg_parent, grid_parent).isel(eta_v = etasliceparent, xi_u = xisliceparent)
sgradmag_surf_parent.name = 'sgradmag'
sgradmag_surf_child = surface_saltgradmag(ds_avg_child, grid_avg_child).isel(eta_v = etaslicechild, xi_u = xislicechild)
sgradmag_surf_child.name = 'sgradmag'

#Compute the surface vorticity
rv_surf_parent = surface_vorticity(ds_avg_parent, grid_parent).isel(eta_v = etasliceparent, xi_u = xisliceparent)
rv_surf_parent.name = 'rvort'
rv_surf_child = surface_vorticity(ds_avg_child, grid_avg_child).isel(eta_v = etaslicechild, xi_u = xislicechild)
rv_surf_child.name = 'rvort'
    
#Compute surface vorticity and strain
divergence_parent = surface_divergence(ds_avg_parent, grid_parent).isel(eta_v = etasliceparent, xi_u = xisliceparent)
divergence_parent.name = 'divergence'
divergence_child = surface_divergence(ds_avg_child, grid_avg_child).isel(eta_v = etaslicechild, xi_u = xislicechild)
divergence_child.name = 'divergence'

strain_parent = surface_strain(ds_avg_parent, grid_parent).isel(eta_v = etasliceparent, xi_u = xisliceparent)
strain_parent.name = 'strain'
strain_child = surface_strain(ds_avg_child, grid_avg_child).isel(eta_v = etaslicechild, xi_u = xislicechild)
strain_child.name = 'strain'

#Compute the rms for each variable 
def rms(y_load):
    '''
    Computes the root-mean square of an Xarray DataArray
Inputs - 
Loaded Xarray DataArray
    '''
    rms = np.sqrt(np.mean(y_load**2))
    return rms 

rms_rv_parent = rms(rv_subset_parent.values)
rms_rv_child = rms(rv_subset_child.values)

rms_divergence_parent = rms(div_subset_parent.values)
rms_divergence_child = rms(div_subset_child.values)

rms_strain_parent = rms(strain_subset_parent.values)
rms_strain_child = rms(strain_subset_child.values)

rms_sgradmag_parent = rms(sgradmag_subset_parent.values)
rms_sgradmag_child = rms(gradmag_subset_child.values)

#Save to numpy arrays 
np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/relvort_surface_parent_rv_rms.npy', rms_rv_parent)
np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/relvort_surface_child_rv_rms.npy', rms_rv_child)

np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/relvort_surface_parent_div_rms.npy', rms_divergence_parent)
np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/relvort_surface_child_div_rms.npy', rms_divergence_child)

np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/relvort_surface_parent_strain_rms.npy', rms_strain_parent)
np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/relvort_surface_child_strain_rms.npy', rms_strain_child)

np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/relvort_surface_parent_sgradmag_rms.npy', rms_sgradmag_parent)
np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/relvort_surface_child_sgradmag_rms.npy', rms_sgradmag_child)