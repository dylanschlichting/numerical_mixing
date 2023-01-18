'''
Computes the diffusive fluxes through the lateral boundaries of a 3D control volume
of salt squared, volume-mean salinity variance, and the cross diffusive flux. 
This is intended to compare with volume-integrated calculations to determine errors associated
with the sloping surface and bottom

See Theory Section of Schlichting et al. for the typesetted definitions.
'''
#Packages 
import numpy as np
import xgcm
from xgcm import Grid
import xarray as xr
import xroms
from xhistogram.xarray import histogram
from glob import glob

xislice = slice(271,404)
etaslice = slice(31,149)

#Open model output
path = glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver1/1hr/ocean_avg_0000*.nc')
# path = glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver1/30min/ocean_avg_0000*.nc')
# path = glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver1/10min/ocean_avg_0000*.nc')
ds = xroms.open_mfnetcdf(path)
ds, grid = xroms.roms_dataset(ds)

def calc_kh(ds):
    '''
Calculates the horizontal eddy diffusivity scaled to 
the grid size
    '''
    dA = ds.dA
    dA_max = (np.sqrt(dA)).max()
    kh_0 = 1.0 #m^2/s
    kh = (kh_0/dA_max)*(np.sqrt(dA))
    return kh

kh_parent = calc_kh(ds)

def horz_s2_diff_vol(ds, etaslice, xislice, grid, k_h):
    '''
Calculates the volume-integrated horizontal salt squared
diffusive flux.
    '''
    ds2dxi, ds2deta = xroms.hgrad(ds.salt**2, grid)
    k_h_u = grid.interp(k_h, 'X')
    k_h_v = grid.interp(k_h, 'Y')

    inner_u = k_h_u*ds2dxi
    inner_u.name = 'inner_u'
    inner_v = k_h_v*ds2deta
    inner_v.name = 'inner_v'

    diff_x, test = xroms.hgrad(inner_u, grid)
    test1, diff_y = xroms.hgrad(inner_v, grid)

    diff_x_rho = grid.interp(diff_x, 'Z')
    diff_y_rho = grid.interp(diff_y, 'Z')

    s2_diff_vol = (((diff_x_rho+diff_y_rho)*ds_avg.dV).isel(eta_rho = etaslice, xi_rho = xislice)).sum(['eta_rho', 'xi_rho', 's_rho'])
    return s2_diff_vol

def horz_sprime2_diff_vol(ds, etaslice, xislice, grid, k_h):
    '''
Calculates the horizontal volume-mean salinity variance 
diffusive flux.
    '''
    dV = ds.dV.isel(eta_rho = etaslice, xi_rho = xislice)
    V = dV.sum(dim = ['eta_rho', 's_rho', 'xi_rho'])
    salt = ds.salt.isel(eta_rho = etaslice, xi_rho = xislice)
    sbar = (1/V)*(salt*dV).sum(dim = ['eta_rho', 'xi_rho','s_rho'])
    sprime = ds.salt-sbar
    sprime.name = 'sprime'

    dsprime2dxi, dsprime2deta = xroms.hgrad(sprime**2, grid)
    k_h_u = grid.interp(k_h, 'X')
    k_h_v = grid.interp(k_h, 'Y')

    inner_u = k_h_u*dsprime2dxi
    inner_u.name = 'inner_u'
    inner_v = k_h_v*dsprime2deta
    inner_v.name = 'inner_v'

    diff_x, test = xroms.hgrad(inner_u, grid)
    test1, diff_y = xroms.hgrad(inner_v, grid)

    diff_x_rho = grid.interp(diff_x, 'Z')
    diff_y_rho = grid.interp(diff_y, 'Z')

    sprime2_diff_vol = (((diff_x_rho+diff_y_rho)*ds_avg.dV).isel(eta_rho = etaslice, xi_rho = xislice)).sum(['eta_rho', 'xi_rho', 's_rho'])
    return sprime2_diff_vol

def horz_sbarsprime_diff_vol(ds, etaslice, xislice, grid, k_h):
    '''
Calculates the boundary horizontal cross-diffusive flux. 
    '''
    dV = ds.dV.isel(eta_rho = etaslice, xi_rho = xislice)
    V = dV.sum(dim = ['eta_rho', 's_rho', 'xi_rho'])
    salt = ds.salt.isel(eta_rho = etaslice, xi_rho = xislice)
    sbar = (1/V)*(salt*dV).sum(dim = ['eta_rho', 'xi_rho','s_rho'])
    sprime = ds.salt-sbar
    sprime.name = 'sprime'

    dsprimedxi, dsprimedeta = xroms.hgrad(sprime, grid)
    k_h_u = grid.interp(k_h, 'X')
    k_h_v = grid.interp(k_h, 'Y')

    inner_u = k_h_u*dsprimedxi
    inner_u.name = 'inner_u'
    inner_v = k_h_v*dsprimedeta
    inner_v.name = 'inner_v'

    diff_x, test = xroms.hgrad(inner_u, grid)
    test1, diff_y = xroms.hgrad(inner_v, grid)

    diff_x_rho = grid.interp(diff_x, 'Z')
    diff_y_rho = grid.interp(diff_y, 'Z')

    sbarsprime_diff_vol = (2*sbar*((diff_x_rho+diff_y_rho)*ds_avg.dV).isel(eta_rho = etaslice, xi_rho = xislice)).sum(['eta_rho', 'xi_rho', 's_rho'])
    return sbarsprime_diff_vol

#Run functions
s2_diff_vol = horz_s2_diff_vol(ds_avg, etaslice, xislice, grid_avg, kh_parent)
sprime2_diff_vol = horz_sprime2_diff_vol(ds_avg, etaslice, xislice, grid_avg, kh_parent)
sbarsprime_diff_vol = horz_sbarsprime_diff_vol(ds_avg, etaslice, xislice, grid_avg, kh_parent)  

#Subset output and save to netcdf files 
dates = np.arange('2010-06-03', '2010-07-14', dtype = 'datetime64[D]') 
for d in range(len(dates)):
    #s2
    s2_horz_diff_parent_sel = s2_diff_vol.sel(ocean_time = str(dates[d]))
    s2_horz_diff_parent_sel.attrs = ''
    path = '/d2/home/dylan/JAMES/revised_submission/budget_recalcs/30min/hdiffusion/s2_hdiffusion_vol_60min_%s.nc' %d
    # path = '/d2/home/dylan/JAMES/revised_submission/budget_recalcs/30min/hdiffusion/s2_hdiffusion_vol_30min_%s.nc' %d
    # path = '/d2/home/dylan/JAMES/revised_submission/budget_recalcs/10min/hdiffusion/s2_hdiffusion_vol_10min_%s.nc' %d
    s2_horz_diff_parent_sel.name = 's2_diff_flux'
    s2_horz_diff_parent_sel.to_netcdf(path, mode = 'w')
    
    #Volume-mean salinity variance
    sprime2_horz_diff_parent_sel = sprime2_diff_vol.sel(ocean_time = str(dates[d]))
    sprime2_horz_diff_parent_sel.attrs = ''
    path = '/d2/home/dylan/JAMES/revised_submission/budget_recalcs/60min/hdiffusion/sprime2_hdiffusion_vol_60min_%s.nc' %d
    # path = '/d2/home/dylan/JAMES/revised_submission/budget_recalcs/30min/hdiffusion/sprime2_hdiffusion_vol_30min_%s.nc' %d
    # path = '/d2/home/dylan/JAMES/revised_submission/budget_recalcs/10min/hdiffusion/sprime2_hdiffusion_vol_10min_%s.nc' %d
    sprime2_horz_diff_parent_sel.name = 'sprime2_diff_flux'
    sprime2_horz_diff_parent_sel.to_netcdf(path, mode = 'w')
    
    #Cross diffusive flux
    sbarsprime_horz_diff_parent_sel = sbarsprime_diff_vol.sel(ocean_time = str(dates[d]))
    sbarsprime_horz_diff_parent_sel.attrs = ''
    path = '/d2/home/dylan/JAMES/revised_submission/budget_recalcs/60min/hdiffusion/sbarsprime_hdiffusion_vol_60min_%s.nc' %d
    # path = '/d2/home/dylan/JAMES/revised_submission/budget_recalcs/30min/hdiffusion/sbarsprime_hdiffusion_vol_30min_%s.nc' %d
    # path = '/d2/home/dylan/JAMES/revised_submission/budget_recalcs/10min/hdiffusion/sbarsprime_hdiffusion_vol_10min_%s.nc' %d
    sbarsprime_horz_diff_parent_sel.name = 'sbarsprime_diff_flux'
    sbarsprime_horz_diff_parent_sel.to_netcdf(path, mode = 'w')
