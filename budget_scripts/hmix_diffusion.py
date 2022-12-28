'''
This script computes the horizontal component of the physical mixing and diffusive flux through the lateral boundaries
for the TXLA native parent model of s^2 and volume-mean salinity variance. 

Notes:
-----
horizontal mixing = $\iiint \kappa_h(\nabla_h s^\prime)^2 \, dV$'
horizontal diffusion = $\iint_{A_l} \, \kappa_h(\nabla_h s^{\prime^2}) \, dA$
Source code for horizontal diffusivity: see 'ini_hmixcoef.F' and 'metrics.F'
$K_h = (K_{h,0}/max(\sqrt{dxdy})(\sqrt{dxdy})$
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

# path = glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver1/1hr/ocean_avg_0000*.nc')
# path = glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver1/30min/ocean_avg_0000*.nc')
path = glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver1/10min/ocean_avg_0000*.nc')
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

kh = calc_kh(ds)

#Horizontal mixing
def chi_horz_hgrad(grid, ds, kh, etaslice, xislice):
    '''
Computes the horizontal component of salinity variance dissipation using the xroms hgrad 
Notes: chi_h = 2 * kappa_h * \nabla_H(s)^2
    '''
    dsaltdxi, dsaltdeta = xroms.hgrad(ds.salt, grid)
    dsdx_rho = grid.interp(xroms.to_rho(dsaltdxi, grid), 'Z')
    dsdy_rho = grid.interp(xroms.to_rho(dsaltdeta, grid), 'Z')

    chih = 2*kh*(dsdx_rho**2+dsdy_rho**2)
    chih_int_hgrad = (chih*ds.dV).isel(eta_rho = etaslice, xi_rho = xislice).sum(['eta_rho', 'xi_rho', 's_rho'])

    return chih_int_hgrad

chih_int_hgrad = chi_horz_hgrad(grid, ds, kh, etaslice, xislice)
print('saving outputs')
dates = np.arange('2010-06-03', '2010-07-14', dtype = 'datetime64[D]') 
for d in range(len(dates)):
    chih_int_sel = chih_int_hgrad.sel(ocean_time = str(dates[d]))
    chih_int_sel.attrs = ''
    # path = '/d2/home/dylan/JAMES/revised_submission/budget_recalcs/30min/hmix/chi_horizontal_30min_%s.nc' %d
    path = '/d2/home/dylan/JAMES/revised_submission/budget_recalcs/10min/hmix/chi_horizontal_10min_%s.nc' %d
    chih_int_sel.name = 'chi_horizontal'
    chih_int_sel.to_netcdf(path, mode = 'w')
    
#Horizontal s^2 diffusion
def horz_s2_diffusion(ds, etaslice, xislice, grid, k_h):    
    dsalt2dxi, dsalt2deta = xroms.hgrad(ds.salt**2, grid)
    #Interpolate to the rho points
    dsalt2dx_rho = grid.interp(dsalt2dxi, 'Z')
    dsalt2dy_rho = grid.interp(dsalt2deta, 'Z')

    k_h_u = grid.interp(k_h, 'X')
    k_h_v = grid.interp(k_h, 'Y')

    dxdz = (ds.dx_v*ds.dz_v)
    dydz = (ds.dy_u*ds.dz_u)

    hdiffusion_u = (k_h_u*dsalt2dx_rho*dydz).isel(eta_rho = etaslice, xi_u = slice(xislice.start-1, xislice.stop))
    hdiffusion_v = (k_h_v*dsalt2dy_rho*dxdz).isel(eta_v = slice(etaslice.start-1, etaslice.stop), xi_rho = xislice)

    hdiffW = (hdiffusion_u).isel(xi_u = 0) #West
    hdiffE = (hdiffusion_u).isel(xi_u = -1) #East
    hdiffN = (hdiffusion_v).isel(eta_v = -1) #North
    hdiffS = (hdiffusion_v).isel(eta_v = 0) #South
    
    #Name individual components to merge into a dataset
    hdiffW.name = 'hdiffW'
    hdiffE.name = 'hdiffE'
    hdiffN.name = 'hdiffN'
    hdiffS.name = 'hdiffS'

    hdiffds = xr.merge([hdiffW, hdiffE, hdiffN, hdiffS], compat='override')
    
    s2_horz_diff = -(hdiffds.hdiffW.sum(['eta_rho', 's_rho'])-hdiffds.hdiffE.sum(['eta_rho', 's_rho']) \
                        +hdiffds.hdiffS.sum(['xi_rho', 's_rho'])-hdiffds.hdiffN.sum(['xi_rho', 's_rho']))
    return s2_horz_diff


#Horizontal s'^2 diffusion
def horz_sprime2_diffusion(ds, etaslice, xislice, grid, k_h):
    dV = ds.dV.isel(eta_rho = etaslice, xi_rho = xislice)
    V = dV.sum(dim = ['eta_rho', 's_rho', 'xi_rho'])
    salt = ds.salt.isel(eta_rho = etaslice, xi_rho = xislice)
    sbar = (1/V)*(salt*dV).sum(dim = ['eta_rho', 'xi_rho','s_rho'])
    sprime = ds.salt-sbar
    sprime2 = sprime**2
    sprime2.name = 'sprime2'
    
    dsaltprime2dxi, dsaltprime2deta = xroms.hgrad(sprime2, grid)
    #Interpolate to the rho points
    dsprime2dx_rho = grid.interp(dsaltprime2dxi, 'Z')
    dsprime2dy_rho = grid.interp(dsaltprime2deta, 'Z')

    k_h_u = grid.interp(k_h, 'X')
    k_h_v = grid.interp(k_h, 'Y')

    dxdz = (ds.dx_v*ds.dz_v)
    dydz = (ds.dy_u*ds.dz_u)

    hdiffusion_u = (k_h_u*dsprime2dx_rho*dydz).isel(eta_rho = etaslice, xi_u = slice(xislice.start-1, xislice.stop))
    hdiffusion_v = (k_h_v*dsprime2dy_rho*dxdz).isel(eta_v = slice(etaslice.start-1, etaslice.stop), xi_rho = xislice)

    hdiffW = (hdiffusion_u).isel(xi_u = 0) #West
    hdiffE = (hdiffusion_u).isel(xi_u = -1) #East
    hdiffN = (hdiffusion_v).isel(eta_v = -1) #North
    hdiffS = (hdiffusion_v).isel(eta_v = 0) #South
    
    #Name individual components to merge into a dataset
    hdiffW.name = 'hdiffW'
    hdiffE.name = 'hdiffE'
    hdiffN.name = 'hdiffN'
    hdiffS.name = 'hdiffS'

    hdiffds = xr.merge([hdiffW, hdiffE, hdiffN, hdiffS], compat='override')
    
    sprime2_horz_diff = -(hdiffds.hdiffW.sum(['eta_rho', 's_rho'])-hdiffds.hdiffE.sum(['eta_rho', 's_rho']) \
                        +hdiffds.hdiffS.sum(['xi_rho', 's_rho'])-hdiffds.hdiffN.sum(['xi_rho', 's_rho']))
    return sprime2_horz_diff

#Parent
sprime2_horz_diff_parent = horz_sprime2_diffusion(ds, etaslice, xislice, grid, kh)
s2_horz_diff_parent = horz_s2_diffusion(ds, etaslice, xislice, grid, kh)

dates = np.arange('2010-06-03', '2010-07-14', dtype = 'datetime64[D]') 
for d in range(len(dates)):
    s2_horz_diff_parent_sel = s2_horz_diff_parent.sel(ocean_time = str(dates[d]))
    s2_horz_diff_parent_sel.attrs = ''
    # path = '/d2/home/dylan/JAMES/revised_submission/budget_recalcs/30min/hdiffusion/s2_hdiffusion_30min_%s.nc' %d
    path = '/d2/home/dylan/JAMES/revised_submission/budget_recalcs/10min/hdiffusion/s2_hdiffusion_10min_%s.nc' %d
    s2_horz_diff_parent_sel.name = 's2_diff_flux'
    s2_horz_diff_parent_sel.to_netcdf(path, mode = 'w')
    
    sprime2_horz_diff_parent_sel = sprime2_horz_diff_parent.sel(ocean_time = str(dates[d]))
    sprime2_horz_diff_parent_sel.attrs = ''
    # path = '/d2/home/dylan/JAMES/revised_submission/budget_recalcs/30min/hdiffusion/sprime2_hdiffusion_30min_%s.nc' %d
    path = '/d2/home/dylan/JAMES/revised_submission/budget_recalcs/10min/hdiffusion/sprime2_hdiffusion_10min_%s.nc' %d
    sprime2_horz_diff_parent_sel.name = 's2_diff_flux'
    sprime2_horz_diff_parent_sel.to_netcdf(path, mode = 'w')
    