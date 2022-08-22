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

path = glob.glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver0/ocean_avg_0000*.nc')
ds_avg_parent = xroms.open_mfnetcdf(path)
ds_avg_parent, grid_parent = xroms.roms_dataset(ds_avg_parent)

path = glob.glob('/d1/shared/TXLA_ROMS/numerical_mixing/nest/ver1/ocean_avg_child_0000*.nc')
ds_avg_child = xroms.open_mfnetcdf(path)
ds_avg_child, grid_avg_child = xroms.roms_dataset(ds_avg_child)

#Corresponding location of the parent model subset. See 'check_grid.ipynb' for more information
xisliceparent = slice(271,404)
etasliceparent = slice(31,149)
xislicechild = slice(8, 677-8)
etaslicechild = slice(8, 602-8)

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
rvortbins = np.linspace(-6,6,150)
divbins = np.linspace(-4,4,150)
strainbins = np.linspace(0,5,150)
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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst. Thanks stack exchange!"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

#Compute the surface salinity gradient magnitude
sgradmag_surf_parent = surface_saltgradmag(ds_avg_parent, grid_parent).isel(eta_v = etasliceparent, xi_u = xisliceparent).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
sgradmag_surf_parent.name = 'sgradmag'
sgradmag_surf_child = surface_saltgradmag(ds_avg_child, grid_avg_child).isel(eta_v = etaslicechild, xi_u = xislicechild).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
sgradmag_surf_child.name = 'sgradmag'

#Create datelist every two hows for subsetting
daterange = list(chunks(ds_avg_child.ocean_time.sel(ocean_time = slice('2010-06-03', '2010-07-13')), 2))
print('Saving sgradmag histograms')

#Compute the surface vorticity
rv_surf_parent = surface_vorticity(ds_avg_parent, grid_parent).isel(eta_v = etasliceparent, xi_u = xisliceparent).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
rv_surf_parent.name = 'rvort'
rv_surf_child = surface_vorticity(ds_avg_child, grid_avg_child).isel(eta_v = etaslicechild, xi_u = xislicechild).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
rv_surf_child.name = 'rvort'

#Compute surface vorticity and strain
divergence_parent = surface_divergence(ds_avg_parent, grid_parent).isel(eta_v = etasliceparent, xi_u = xisliceparent).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
divergence_parent.name = 'divergence'
divergence_child = surface_divergence(ds_avg_child, grid_avg_child).isel(eta_v = etaslicechild, xi_u = xislicechild).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
divergence_child.name = 'divergence'

strain_parent = surface_strain(ds_avg_parent, grid_parent).isel(eta_v = etasliceparent, xi_u = xisliceparent).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
strain_parent.name = 'strain'
strain_child = surface_strain(ds_avg_child, grid_avg_child).isel(eta_v = etaslicechild, xi_u = xislicechild).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
strain_child.name = 'strain'

#Compute statistics - parent
# rvort_mean = rv_surf_parent.mean().compute()
# rvort_mean_vals = rvort_mean.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/relvort_surface_parent_mean.npy', rvort_mean_vals)

# rvort_median = rv_surf_parent.median().compute()
# rvort_median_vals = rvort_median.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/relvort_surface_parent_median.npy', rvort_median_vals)

# rvort_std = rv_surf_parent.std().compute()
# rvort_std_vals = rvort_std.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/relvort_surface_parent_std.npy', rvort_std_vals)

# divergence_mean = divergence_parent.mean().compute()
# divergence_mean_vals = divergence_mean.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/divergence_surface_parent_mean.npy', divergence_mean_vals)

# divergence_median = divergence_parent.median().compute()
# divergence_median_vals = divergence_median.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/divergence_surface_parent_median.npy', divergence_median_vals)

# divergence_std = divergence_parent.std().compute()
# divergence_std_vals = divergence_std.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/divergence_surface_parent_std.npy', divergence_std_vals)

# strain_mean = strain_parent.mean().compute()
# strain_mean_vals = strain_mean.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/strain_surface_parent_mean.npy', strain_mean_vals)

# strain_median = strain_parent.median().compute()
# strain_median_vals = strain_median.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/strain_surface_parent_median.npy', strain_median_vals)

# strain_std = strain_parent.std().compute()
# strain_std_vals = strain_std.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/strain_surface_parent_std.npy', strain_std_vals)

# sgradmag_mean = sgradmag_surf_parent.mean().compute()
# sgradmag_mean_vals = sgradmag_mean.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/sgradmag_surface_parent_mean.npy', sgradmag_mean_vals)

# sgradmag_median = sgradmag_surf_parent.median().compute()
# sgradmag_median_vals = sgradmag_median.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/sgradmag_surface_parent_median.npy', sgradmag_median_vals)

# sgradmag_std = sgradmag_surf_parent.std().compute()
# sgradmag_std_vals = sgradmag_std.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/sgradmag_surface_parent_std.npy', sgradmag_std_vals)

# #Child --------
# rvort_mean = rv_surf_child.mean().compute()
# rvort_mean_vals = rvort_mean.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/relvort_surface_child_mean.npy', rvort_mean_vals)

# rvort_median = rv_surf_child.median().compute()
# rvort_median_vals = rvort_median.values()
# np.save('/d2/home/dylan/JAMES/histogram_outputs/stats/relvort_surface_child_median.npy', rvort_median_vals)

# rvort_std = rv_surf_child.std().compute()
# rvort_std_vals = rvort_std.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/relvort_surface_child_std.npy', rvort_std_vals)

# divergence_mean = divergence_child.mean().compute()
# divergence_mean_vals = divergence_mean.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/divergence_surface_child_mean.npy', divergence_mean_vals)

# divergence_median = divergence_child.median().compute()
# divergence_median_vals = divergence_median.values()
# np.save('/d2/home/dylan/JAMES/histogram_outputs/stats/divergence_surface_child_median.npy', divergence_median_vals)

# divergence_std = divergence_child.std().compute()
# divergence_std_vals = divergence_std.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/divergence_surface_child_std.npy', divergence_std_vals)

# strain_mean = strain_child.mean().compute()
# strain_mean_vals = strain_mean.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/strain_surface_child_mean.npy', strain_mean_vals)

# strain_median = strain_child.median().compute()
# strain_median_vals = strain_median.values()
# np.save('/d2/home/dylan/JAMES/histogram_outputs/stats/strain_surface_child_median.npy', strain_median_vals)

# strain_std = strain_child.std().compute()
# strain_std_vals = strain_std.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/strain_surface_child_std.npy', strain_std_vals)

# sgradmag_mean = sgradmag_surf_child.mean().compute()
# sgradmag_mean_vals = sgradmag_mean.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/sgradmag_surface_child_mean.npy', sgradmag_mean_vals)

# sgradmag_median = sgradmag_surf_child.median().compute()
# sgradmag_median_vals = sgradmag_median.values()
# np.save('/d2/home/dylan/JAMES/histogram_outputs/stats/sgradmag_surface_child_median.npy', sgradmag_median_vals)

# sgradmag_std = sgradmag_surf_child.std().compute()
# sgradmag_std_vals = sgradmag_std.values
# np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/sgradmag_surface_child_std.npy', sgradmag_std_vals)

# Additional calculation method --------
# Xarray's median function does not work for more than one dimension, it raises an error code. The data are 
# small enough to convert to numpy arrays and then use their in-house functions to concatenate. So we will do that instead. 
newdims_parent = len(rv_surf_parent.ocean_time)*len(rv_surf_parent.xi_u)*len(rv_surf_parent.eta_v)

rvnew = np.reshape(np.array(rv_surf_parent), (newdims_parent), order = 'F')
rvmedian = np.median(rvnew)
np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/relvort_surface_parent_median.npy', rvmedian)

divnew = np.reshape(np.array(divergence_parent), (newdims_parent), order = 'F')
divmedian = np.median(divnew)
np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/divergence_surface_parent_median.npy', divmedian)

sgradmag_surfnew = np.reshape(np.array(sgradmag_surf_parent), (newdims_parent), order = 'F')
sgradmagmedian = np.median(sgradmag_surfnew)
np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/sgradmag_surface_parent_median.npy', sgradmagmedian)

strainnew = np.reshape(np.array(strain_parent), (newdims_parent), order = 'F')
strainmedian = np.median(strainnew)
np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/strain_surface_parent_median.npy', strainmedian)

#Child 
newdims_child = len(rv_surf_child.ocean_time)*len(rv_surf_child.xi_u)*len(rv_surf_child.eta_v)

rvnew_child = np.reshape(np.array(rv_surf_child), (newdims_child), order = 'F')
rvmedian_child = np.median(rvnew_child)
np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/relvort_surface_child_median.npy', rvmedian_child)

divnew_child = np.reshape(np.array(divergence_child), (newdims_child), order = 'F')
divmedian_child = np.median(divnew_child)
np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/divergence_surface_child_median.npy', divmedian_child)

sgradmag_surfnew_child = np.reshape(np.array(sgradmag_surf_child), (newdims_child), order = 'F')
sgradmagmedian_child = np.median(sgradmag_surfnew_child)
np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/sgradmag_surface_child_median.npy', sgradmagmedian_child)

strainnew_child = np.reshape(np.array(strain_child), (newdims_child), order = 'F')
strainmedian_child = np.median(strainnew_child)
np.save('/d2/home/dylan/JAMES/histogram_outputs/surface/stats/strain_surface_child_median.npy', strainmedian_child)
