#Packages 
import numpy as np
import xgcm
from xgcm import Grid
import xarray as xr
import xroms
from datetime import datetime
import glob
from xhistogram.xarray import histogram

path = glob.glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver1/1hr/ocean_avg_0000*.nc')
ds_avg = xroms.open_mfnetcdf(path)
ds_avg, grid_avg = xroms.roms_dataset(ds_avg)

path = glob.glob('/d1/shared/TXLA_ROMS/numerical_mixing/nest/ver1/ocean_avg_child_0000*.nc')
ds_avg_child = xroms.open_mfnetcdf(path)
ds_avg_child, grid_avg_child = xroms.roms_dataset(ds_avg_child)

#Match the times of the parent and child models. Drop the first timestep after
#each restart, then match parent / child. 
timedrop = [np.datetime64('2010-06-18T18:30:00.000000000'), 
            np.datetime64('2010-06-19T18:30:00.000000000'), 
            np.datetime64('2010-07-09T18:30:00.000000000')]

ds_avg_child = ds_avg_child.where((ds_avg_child.ocean_time!= timedrop[0])
                                & (ds_avg_child.ocean_time!= timedrop[1])
                                & (ds_avg_child.ocean_time!= timedrop[2]),
                                   drop=True)

ds_avg = ds_avg.where(ds_avg_child.ocean_time==ds_avg.ocean_time)

def rvort(ds, grid):
    '''
Computes the relative vertical vorticity,for the entire water column normalized by the Coriolis
frequence. All quantities are either computed on or interpolated linearly to the
psi points horizontally and to the rho points vertically.
    '''
    rvort = xroms.relative_vorticity(ds.u, ds.v, grid)
    rvort_srho = grid.interp(rvort, 'Z', boundary = 'extend')
    zeta_f = xroms.to_rho(rvort_srho, grid)/ds.f
    
    return zeta_f

def thomas_angle(ds, grid):
    # density and buoyancy
    sigma_theta = xroms.potential_density(ds.temp, ds.salt, z=0)
    b = xroms.buoyancy(sigma_theta, rho0=1025.0)

    # buoyancy gradients - on the s and xi/eta_rho points
    dbdx, dbdy = xroms.hgrad(b, grid, hcoord = 'rho', scoord='rho')
    delb2 = dbdx**2 + dbdy**2
    N2 = grid.interp(grid.derivative(b, 'Z', boundary='extend'), 'Z', boundary = 'extend')

    # Vertical vorticity - on the s and xi/eta_rho points
    rvort = xroms.relative_vorticity(ds.u, ds.v, grid)
    rvort_srho = grid.interp(rvort, 'Z', boundary = 'extend')
    zeta = xroms.to_rho(rvort_srho, grid)
    
    # Thomas angle
    phi_T = np.arctan2(-delb2, (ds.f**2 * N2))*180.0/np.pi
    phi_T.coords['lon_rho'] = ds.coords['lon_rho']
    phi_T.coords['lat_rho'] = ds.coords['lat_rho']
    phi_T.coords['z_rho'] = ds.coords['z_rho']
    phi_T.name = r'Thomas angle, $\phi_T$'

    # critical angle
    phi_c = np.arctan2(-(ds.f + zeta) ,ds.f)*180.0/np.pi
    
    return phi_T, phi_c

#Parent model slices
xislice = slice(271,404)
etaslice = slice(31,149)

#Child model slices
xislice_child = slice(8, 677-8)
etaslice_child = slice(8, 602-8)

phi_T_parent, phi_c_parent = thomas_angle(ds_avg, grid_avg)
phi_T_child, phi_c_child = thomas_angle(ds_avg_child, grid_avg_child)

#Compute vorticity
zeta_f_parent = rvort(ds_avg, grid_avg)
zeta_f_child = rvort(ds_avg_child, grid_avg_child)

#Separate out positive and negative values of thomas angle. This could be adjusted for zeta/f < 1 or zeta/f > 1 for sensitivity studies
phi_t_parent_pos = phi_T_parent.where(zeta_f_parent>0).isel(eta_rho = etaslice, xi_rho = xislice).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
phi_t_parent_pos.attrs = ''
phi_t_parent_pos.name = 'phi_t_pos'

phi_t_parent_neg = phi_T_parent.where(zeta_f_parent<0).isel(eta_rho = etaslice, xi_rho = xislice).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
phi_t_parent_neg.attrs = ''
phi_t_parent_neg.name = 'phi_t_neg'

phi_t_child_pos = phi_T_child.where(zeta_f_child>0).isel(eta_rho = etaslice_child, xi_rho = xislice_child).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
phi_t_child_pos.attrs = ''
phi_t_child_pos.name = 'phi_t_pos'

phi_t_child_neg = phi_T_child.where(zeta_f_child<0).isel(eta_rho = etaslice_child, xi_rho = xislice_child).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
phi_t_child_neg.attrs = ''
phi_t_child_neg.name = 'phi_t_neg'

phitbins = np.linspace(-180.5,0.5, 182) #range for thomas angle

dV = ds_avg.dV.isel(eta_rho = etaslice, xi_rho = xislice).sel(ocean_time = slice('2010-06-03', '2010-07-13'))

phi_t_parent_pos_hist = histogram(phi_t_parent_pos, bins = [phitbins], weights = dV, density = True)
phi_t_parent_pos_hist.name = 'phi_t_pos'
phi_t_parent_pos_hist.to_netcdf('/d2/home/dylan/JAMES/initial_submission/phi_t_parent_pos_hist_dV.nc')

phi_t_parent_neg_hist = histogram(phi_t_parent_neg, bins = [phitbins], weights = dV, density = True)
phi_t_parent_neg_hist.name = 'phi_t_neg'
phi_t_parent_neg_hist.to_netcdf('/d2/home/dylan/JAMES/initial_submission/phi_t_parent_neg_hist_dV.nc')

# ----------------
# The cluster can't handle the child model all at once, break it into chunks of two - six hours (2 timesteps) 
# and save those like the whole histograms 

dV_child = ds_avg_child.isel(eta_rho = etaslice_child, xi_rho = xislice_child).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
def chunks(lst, n):
    """Yield successive n-sized chunks from lst. Thanks stack exchange!"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
daterange = list(chunks(ds_avg_child.ocean_time.sel(ocean_time = slice('2010-06-03', '2010-07-14')), 4))
for d in range(len(daterange)):
    phi_t_child_pos_hist = histogram(phi_t_child_pos.sel(ocean_time = daterange[d]), bins = [phitbins], weights = dV_child, density = True)
    phi_t_child_pos_hist.name = 'phi_t_pos'
    path = '/d2/home/dylan/JAMES/initial_submission/thomas_angle/phi_t_child_pos_hist_dV_%s.nc' %d
    phi_t_child_pos_hist.to_netcdf(path)
    
    phi_t_child_neg_hist = histogram(phi_t_child_neg.sel(ocean_time = daterange[d]), bins = [phitbins], weights = dV_child, density = True)
    phi_t_child_neg_hist.name = 'phi_t_neg'
    path = '/d2/home/dylan/JAMES/initial_submission/thomas_angle/phi_t_child_neg_hist_dV_%s.nc' %d
    phi_t_child_neg_hist.to_netcdf(path)

# Compute histograms that are NOT weighted by dV.     
phi_t_parent_pos = phi_T_parent.where(zeta_f_parent>0).isel(eta_rho = etaslice, xi_rho = xislice).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
phi_t_parent_pos.attrs = ''
phi_t_parent_pos.name = 'phi_t_pos'

phi_t_parent_neg = phi_T_parent.where(zeta_f_parent<0).isel(eta_rho = etaslice, xi_rho = xislice).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
phi_t_parent_neg.attrs = ''
phi_t_parent_neg.name = 'phi_t_neg'

phi_t_child_pos = phi_T_child.where(zeta_f_child>0).isel(eta_rho = etaslice_child, xi_rho = xislice_child).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
phi_t_child_pos.attrs = ''
phi_t_child_pos.name = 'phi_t_pos'

phi_t_child_neg = phi_T_child.where(zeta_f_child<0).isel(eta_rho = etaslice_child, xi_rho = xislice_child).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
phi_t_child_neg.attrs = ''
phi_t_child_neg.name = 'phi_t_neg'

phitbins = np.linspace(-180.5,0.5, 182)

phi_t_parent_pos_hist = histogram(phi_t_parent_pos, bins = [phitbins], density = True)
phi_t_parent_pos_hist.name = 'phi_t_pos'
phi_t_parent_pos_hist.to_netcdf('/d2/home/dylan/JAMES/initial_submission/phi_t_parent_pos_hist1.nc')

phi_t_parent_neg_hist = histogram(phi_t_parent_neg, bins = [phitbins], density = True)
phi_t_parent_neg_hist.name = 'phi_t_neg'
phi_t_parent_neg_hist.to_netcdf('/d2/home/dylan/JAMES/initial_submission/phi_t_parent_neg_hist1.nc')

phi_t_child_pos_hist = histogram(phi_t_child_pos, bins = [phitbins], density = True)
phi_t_child_pos_hist.name = 'phi_t_pos'
phi_t_child_pos_hist.to_netcdf('/d2/home/dylan/JAMES/initial_submission/phi_t_child_pos_hist1.nc')

phi_t_child_neg_hist = histogram(phi_t_child_neg, bins = [phitbins], density = True)
phi_t_child_neg_hist.name = 'phi_t_neg'
phi_t_child_neg_hist.to_netcdf('/d2/home/dylan/JAMES/initial_submission/phi_t_child_neg_hist1.nc')

# ----------------
# If the cluster can't handle the child model, break it into chunks of two hours (2 timesteps) 
# and save those like the whole histograms 

def chunks(lst, n):
    """Yield successive n-sized chunks from lst. Thanks stack exchange."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
daterange = list(chunks(ds_avg_child.ocean_time.sel(ocean_time = slice('2010-06-03', '2010-07-14')), 4))
for d in range(len(daterange)):
    phi_t_child_pos_hist = histogram(phi_t_child_pos.sel(ocean_time = daterange[d]), bins = [phitbins], density = True)
    phi_t_child_pos_hist.name = 'phi_t_pos'
    path = '/d2/home/dylan/JAMES/initial_submission/thomas_angle/phi_t_child_pos_hist1_%s.nc' %d
    phi_t_child_pos_hist.to_netcdf(path)
    
    phi_t_child_neg_hist = histogram(phi_t_child_neg.sel(ocean_time = daterange[d]), bins = [phitbins], density = True)
    phi_t_child_neg_hist.name = 'phi_t_neg'
    path = '/d2/home/dylan/JAMES/initial_submission/thomas_angle/phi_t_child_neg_hist1_%s.nc' %d
    phi_t_child_neg_hist.to_netcdf(path)
