import os
import re
import numpy as np
import xarray as xr
import xesmf as xe
import cftime
import gsw
import pandas as pd
from xgcm import Grid
from xgcm.autogenerate import generate_grid_ds


def dll_dist(dlon, dlat, lon, lat):
    """Converts lat/lon differentials into distances in meters using the WGS84 ellipsoid.
   
    PARAMETERS
    ----------
    dlon : xarray.DataArray longitude differentials
    dlat : xarray.DataArray latitude differentials
    lon  : xarray.DataArray longitude values
    lat  : xarray.DataArray latitude values
   
    RETURNS
    -------
    dx  : xarray.DataArray distance inferred from dlon
    dy  : xarray.DataArray distance inferred from dlat
    """
    # WGS84 ellipsoid parameters
    a = 6378137.0  # semi-major axis (equatorial radius) in meters
    f = 1 / 298.257223563  # flattening
    b = a * (1 - f)
    # Convert latitude and longitude to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    # Calculate the meridional radius of curvature (distance for 1 degree latitude)
    R_lat = (a * (1 - f)) / (1 - (2 * f - f ** 2) * np.sin(lat_rad) ** 2) ** (3 / 2)

    
    # Calculate the prime vertical radius of curvature (distance for 1 degree longitude)
    R_lon = a / (1 - (2 * f - f ** 2) * np.sin(lat_rad) ** 2) ** (1 / 2)
  
    # Calculate distances
    dx = dlon * R_lon * np.cos(lat_rad)
    dy = dlat * R_lat
    
    return dx, dy


def calculate_and_append_curl(files_taux,files_tauy, output_dir):
    
    for i in range(len(files_taux)):
        fl_u=files_taux[i]
        fl_v=files_tauy[i]
        
        try:
            
            output_file = os.path.join(output_dir, os.path.basename(fl_u).replace('taux', 'curl'))
        
        # Check if the output file already exists
            if os.path.exists(output_file):
                print(f"Skipping {fl_u}: Curl file already exists.")
                continue

            ds_u = xr.open_dataset(fl_u)
            ds_v = xr.open_dataset(fl_v)

            ds=ds_u
            ds['sometauy']=ds_v.sometauy

            
            # Check if the required variables are present
            #if 'tauu' not in ds or 'tauv' not in ds:
            #    print(f"Skipping {fl}: Required variables not found.")
            #    continue

            # Generate grid dataset for xgcm
            print(f" processing file {fl_u}")
            lat_2d = np.tile(ds.lat.values[:, np.newaxis], (1, len(ds.lon.values)))
            ds_full = generate_grid_ds(ds, {'Y': 'lat', 'X': 'lon'})
            grid = Grid(ds_full, periodic=['X'])
            
            print('grid generated')

            # Calculate grid differences
            dlong = grid.diff(ds_full.lon, 'X', boundary='extend', fill_value=np.nan)
            dlatg = grid.diff(ds_full.lat, 'Y', boundary='extrapolate', fill_value=np.nan)
            dlong[0] = dlong[1]  # Correct the first row if necessary
            
            print('diff generated')

            # Calculate distances
            dx, dy = dll_dist(dlong, dlatg, ds.lon, ds.lat)
            
            
            print('dist generated')

            # Compute derivatives
            du_dy = grid.diff(ds.sozotaux, 'Y', boundary='fill', fill_value=np.nan) / dy
            dv_dx = grid.diff(ds.sometauy, 'X') / dx
            
            print('gradient generated')

            # Align the lat and lon for difference results
            du_dy['lat_left'] = dv_dx['lat'].values
            dv_dx['lon_left'] = du_dy['lon'].values
            du_dy = np.squeeze(du_dy.isel(lat=0).drop('lat'))

            du_dy = du_dy.rename({'lat_left': 'lat'})
            dv_dx = dv_dx.rename({'lon_left': 'lon'})

            # Calculate curl: dVdx - dUdy
            curl = dv_dx - du_dy
            curl = curl.rename('curl')
            
            print('curl generated')

            curl.to_netcdf(output_file)
            print(f"Saved curl to {output_file}.")

        except Exception as e:
            print(f"Error processing {fl_u}: {e}")
            


def calculate_SSDflux_ERA5_ORAS5_mfdataset(ERA5_fluxes_path, ORAS5_fluxes_path, ERA5_sst_path,ORAS5_sos_path, output_path):
    # Required variables
    required_flux_vars = ['sohefldo']
    required_era5_flux_vars=['e', 'tp']  # ERA5 fluxes
    required_sst_var = 'sst'  # ERA5 Sea Surface Temperature
    required_sos_var = 'sosaline'  # ORAS5 Sea Surface Salinity
    
    # Time range for processing
    start_date = "1970-01-01"
    end_date = "2023-12-31"
    
    # Open multiple files using xarray's open_mfdataset
    flux_files = sorted([os.path.join(ERA5_fluxes_path, f) for f in os.listdir(ERA5_fluxes_path) if f.endswith('.nc')])
    sst_files = sorted([os.path.join(ERA5_sst_path, f) for f in os.listdir(ERA5_sst_path) if f.endswith('.nc')])
    sos_files = sorted([os.path.join(ORAS5_sos_path, f) for f in os.listdir(ORAS5_sos_path) if f.endswith('.nc')])
    hflux_files = sorted([os.path.join(ORAS5_fluxes_path, f) for f in os.listdir(ORAS5_fluxes_path) if f.endswith('.nc')])
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    try:
        print("Opening ERA5 fluxes...")
        flux_ds = xr.open_mfdataset(flux_files, combine="by_coords", chunks={"valid_time": 12},preprocess=lambda ds: ds.drop_vars('expver') if 'expver' in ds.variables else ds, coords='minimal').sel(valid_time=slice(start_date, end_date))

        print("Opening ERA5 SST...")
        sst_ds = xr.open_mfdataset(sst_files, combine="by_coords", chunks={"valid_time": 12}, preprocess=lambda ds: ds.drop_vars('expver') if 'expver' in ds.variables else ds, coords='minimal').sel(valid_time=slice(start_date, end_date))
        
        print("Opening ORAS5 SOS...")
        sos_ds = xr.open_mfdataset(sos_files, combine="by_coords", chunks={"time_counter": 12}, coords='minimal').sel(time_counter=slice(start_date, end_date))
        
        print("Opening ORAS5 hflux...")
        hflux_ds = xr.open_mfdataset(hflux_files, combine="by_coords", chunks={"time_counter": 12}, coords='minimal').sel(time_counter=slice(start_date, end_date))

        
        # Extract variables
        sos = sos_ds[required_sos_var]  # ORAS5 salinity (PSU)
        sst = sst_ds[required_sst_var] - 273.15  # ERA5 temperature (K to °C)
        hflux = hflux_ds[required_flux_vars]
        
        sos['time_counter']=sst.valid_time.values
        flux_ds['valid_time']=sst.valid_time.values
        hflux['time_counter']=sst.valid_time.values
        sos=sos.rename({'time_counter':'valid_time'})
        hflux=hflux.rename({'time_counter':'valid_time'})
        # Convert accumulated monthly heat fluxes to rates (W/m²)
        time = pd.to_datetime(flux_ds['valid_time'].values)
        days_in_month = time.days_in_month 
        
        evaporation_flux = flux_ds['e'] / (24 * 3600)  # Evaporation (kg/m²/s) (multiplied it by water density)
        precipitation_flux = flux_ds['tp'] / (24 * 3600)  # Precipitation (kg/m²/s) (multiplied it by water density)
        
        
        # Total heat flux (W/m²)
        hfds = hflux
        
        # Convert PSU to g/kg for salinity
        SA = sos  # Assuming 35 PSU ≈ 35 g/kg
        
        # Density of seawater (surface pressure, 0 dbar)
        rho = gsw.density.rho(SA, sst, 0)
        
        # Thermal contribution (Ft)
        alpha = gsw.alpha(SA, sst, 0)  # Thermal expansion coefficient (1/degC)
        
        Ft = - alpha * (hfds / 3850)  # Thermal contribution to SSDflux
        
        # Salinity contribution (Fs)
        beta = gsw.beta(SA, sst, 0)  # Haline contraction coefficient (1/(g/kg))
        Fs = rho * beta * SA * ((-evaporation_flux - precipitation_flux) / (1 - (SA/1000)))  # Haline contribution
        
        # Total SSDflux (Frho)
        Frho = Ft + Fs
        
        print(np.shape(Ft.sohefldo.values))
        # Create output dataset
        output_ds = xr.Dataset(
            {
                "Ft": (("time", "lat", "lon"), Ft.sohefldo.values, {"units": "kg/m^2/s", "description": "Thermal contribution to SSDflux"}),
                "Fs": (("time", "lat", "lon"), Fs.values, {"units": "kg/m^2/s", "description": "Salinity contribution to SSDflux"}),
                "Frho": (("time", "lat", "lon"), Frho.sohefldo.values, {"units": "kg/m^2/s", "description": "Total SSDflux"})
            },
            coords={
                "time": flux_ds["valid_time"].values,
                "lat": flux_ds["lat"],
                "lon": flux_ds["lon"]
            },
            attrs={
                "description": "Thermal, salinity, and total SSDflux contributions computed from ERA5 and ORAS5 data."
            }
        )
        
        # Save output dataset
        output_filepath = os.path.join(output_path, "SSDflux_ERA5_ORAS5.nc")
        output_ds.to_netcdf(output_filepath)
        print(f"Saved SSDflux outputs to {output_filepath}")
    
    #except Exception as e:
    #    print(f"Error during processing: {e}")
    
    finally:
        # Ensure datasets are closed
        if 'flux_ds' in locals():
            flux_ds.close()
        if 'sst_ds' in locals():
            sst_ds.close()
        if 'sos_ds' in locals():
            sos_ds.close()
            


# Example usage:
# calculate_SSDflux_ERA5_ORAS5_mfdataset(ERA5_fluxes_path, ERA5_sst_path, ORAS5_sos_path, output_path)


def calculate_and_append_wsm(files_taux,files_tauy, output_dir):
    
    for i in range(len(files_taux)):
        fl_u=files_taux[i]
        fl_v=files_tauy[i]
        
        try:
            
            output_file = os.path.join(output_dir, os.path.basename(fl_u).replace('taux', 'wsm'))
        
        # Check if the output file already exists
            if os.path.exists(output_file):
                print(f"Skipping {fl_u}: Curl file already exists.")
                continue

            ds_u = xr.open_dataset(fl_u)
            ds_v = xr.open_dataset(fl_v)

            ds=ds_u
            ds['sometauy']=ds_v.sometauy

            
            # Check if the required variables are present
            #if 'tauu' not in ds or 'tauv' not in ds:
            #    print(f"Skipping {fl}: Required variables not found.")
            #    continue

            wsm = np.sqrt(ds.sozotaux**2 + ds.sometauy**2)
            wsm= wsm.rename('wsm')
            
            print('wsm generated')

            wsm.to_netcdf(output_file)
            print(f"Saved wsn to {output_file}.")

        except Exception as e:
            print(f"Error processing {fl_u}: {e}")
            


            



