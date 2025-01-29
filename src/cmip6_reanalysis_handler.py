import os
import re
import numpy as np
import xarray as xr
import xesmf as xe
import cftime


def get_model_folders(base_dir, year=1990):
    """
    Returns a list of model folders where the earliest file starts from or before a given year.

    Parameters:
    - base_dir (str): The base directory containing model folders.
    - year (int): The cutoff year (default is 1990).

    Returns:
    - list: A list of model folder names meeting the criteria.
    """
    valid_models = []
    cutoff_date = int(f"{year}01")  # Convert year to YYYYMM format (January of the year)

    # Loop through all folders in the base directory
    for model in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model)
        if os.path.isdir(model_path):  # Ensure it's a directory
            dates = []
            
            # Loop through all files in the model folder
            for file in os.listdir(model_path):
                if file.endswith(".nc"):
                    # Extract date ranges using a regex
                    match = re.search(r"(\d{6})-(\d{6})\.nc$", file)
                    if match:
                        dates.append(match.group(1))  # Start date
                        dates.append(match.group(2))  # End date

            # Sort dates and get the first date
            if dates:
                dates = sorted(dates)
                first_date = dates[0]
                if int(first_date) <= cutoff_date:  # Include folders starting on or before the cutoff
                    valid_models.append(model)

    return valid_models


def regrid_models(base_dir, selected_models):
    """
    Concatenate and regrid datasets from selected folders.

    Parameters:
    - base_dir (str): Path to the base directory containing model folders.
    - selected_models (list): List of model folder names to process.

    Returns:
    - dict: A dictionary where keys are model names and values are regridded xarray Datasets.
    """
    # Define the target grid

    regridded_data = {}  # Dictionary to store regridded models

    for model_name in selected_models:
        model_path = os.path.join(base_dir, model_name)
        print(f"Processing model: {model_name}")

        # Concatenate all files in the folder into a single xarray.Dataset
        model_files = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith(".nc")]
        model_ds = xr.open_mfdataset(model_files, concat_dim='time',combine='nested')

        # Regrid the dataset


        # Add regridded dataset to the dictionary
        
        rg_model=regrid_models_simple(model_ds)
        regridded_data[model_name] = rg_model

    return regridded_data

def regrid_models_2(processed_datasets, output_dir='/Volumes/Thalassa/CMIP6_SHDR_RG_360'):
    """
    Concatenate and regrid datasets from selected folders.
    - dict: A dictionary where keys are model names and values are regridded xarray Datasets.
    """
    
    os.makedirs(output_dir, exist_ok=True)

    regridded_data = {}  # Dictionary to store regridded models


    for model_name in list(processed_datasets.keys()):
        
        output_file = os.path.join(output_dir, f"{model_name}_SHDR.nc")
        
        if os.path.exists(output_file):
            print(f"model {model_name} already exists")
            continue
        
        print(f"Processing model: {model_name}")
        
        model_ds=processed_datasets[model_name]

        # Add regridded dataset to the dictionary
        rg_model=regrid_models_simple(model_ds)
        rg_model.to_netcdf(output_file)
        regridded_data[model_name] = rg_model

    return regridded_data

def regrid_models_simple(model_ds, lat_rg=np.arange(-89.5, 90, 1), lon_rg=np.arange(0.5, 360, 1), method="bilinear"):
    
    grid_out = xr.Dataset(
        {
            "lat": (["lat"], lat_rg, {"units": "degrees_north"}),
            "lon": (["lon"], lon_rg, {"units": "degrees_east"}),
        }
    )

    regridder_model = xe.Regridder(model_ds, grid_out, method, periodic=True, ignore_degenerate=True)
    regridded_model_ds = regridder_model(model_ds, keep_attrs=True)

    return regridded_model_ds

def cftime_to_numpy(cftime_obj):
    # Extract components from cftime object
    year = cftime_obj.year
    month = cftime_obj.month
    day = cftime_obj.day
    hour = cftime_obj.hour
    minute = cftime_obj.minute
    second = cftime_obj.second
    microsecond = cftime_obj.microsecond
    
    # Create a string representation and convert it to numpy.datetime64
    datetime_str = f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}.{microsecond:06d}"
    return np.datetime64(datetime_str)

def merge_netcdf_by_time(folder_path, file_extension=".nc", chunks=None, preprocess=None):
    """
    Opens and merges multiple NetCDF files in a folder by time using xarray.

    Parameters:
    - folder_path (str): Path to the folder containing NetCDF files.
    - file_extension (str): File extension to filter files (default is ".nc").
    - chunks (dict, optional): Dictionary for Dask chunking (e.g., {"time": 100}).
    - preprocess (callable, optional): Function to preprocess datasets before merging.

    Returns:
    - xarray.Dataset: A merged dataset containing data from all NetCDF files.
    """
    # List all NetCDF files in the folder
    file_list = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(file_extension)
    ]

    if not file_list:
        raise ValueError(f"No files with extension '{file_extension}' found in {folder_path}")

    # Open and merge NetCDF files
    ds = xr.open_mfdataset(
        file_list, 
        combine="by_coords", 
        chunks=chunks, 
        preprocess=preprocess
    )

    return ds

def handle_time_coordinates(model_ds):
    """Corrige las coordenadas de tiempo en el dataset."""
    if isinstance(model_ds.time.values[0], cftime.DatetimeNoLeap):
        model_ds['time'] = [cftime_to_numpy(t) for t in model_ds['time'].values]

    if isinstance(model_ds.indexes['time'][0], cftime.Datetime360Day):
        model_ds = model_ds.assign_coords(time=model_ds.indexes['time'].to_datetimeindex())

    if len(np.unique(model_ds['time'])) < len(model_ds['time']):
        _, index = np.unique(model_ds['time'], return_index=True)
        unique_mask = np.isin(np.arange(len(model_ds['time'])), index)
        model_ds = model_ds.isel(time=unique_mask)

    model_ds = model_ds.sortby('time')
    return model_ds

def adjust_longitudes(model_ds):
    """Ajusta las longitudes al rango -180 a 180 si es necesario."""
    if "lon" in model_ds.coords:
        lon_min = model_ds.lon.min().values
        if lon_min > 0:  # Si están en el rango 0 a 360
            # Usar assign_coords para ajustar las longitudes
            model_ds = model_ds.assign_coords(lon=((model_ds.lon + 180) % 360) - 180)
    else:
        raise ValueError("Coordinate 'lon' is not in the dataset.")
    return model_ds

def adjust_longitudes_and_latitudes(model_ds):
    """Ajusta las longitudes y valida las latitudes si es necesario."""
    coords_to_update = {}
    
    # Ajustar longitudes al rango -180 a 180
    if "lon" in model_ds.coords:
        lon_min = model_ds.lon.min().values
        if lon_min > 0:  # Si están en el rango 0 a 360
            coords_to_update["lon"] = ((model_ds.lon + 180) % 360) - 180
    else:
        raise ValueError("Coordinate 'lon' is not in the dataset.")
    
    # Validar que 'lat' esté presente
    if "lat" not in model_ds.coords:
        raise ValueError(" Coordinate 'lat' is not in the dataset.")
    
    # Aplicar los cambios acumulados usando assign_coords
    if coords_to_update:
        model_ds = model_ds.assign_coords(coords_to_update)

    return model_ds


def detect_spatial_dimensions(model_ds):
    """Detecta las dimensiones espaciales del dataset."""
    if {"lat", "lon"}.issubset(model_ds.dims):
        spatial_dims = ("lat", "lon")
    elif {"j", "i"}.issubset(model_ds.dims):
        spatial_dims = ("j", "i")
    elif {"y", "x"}.issubset(model_ds.dims):
        spatial_dims = ("y","x")
    else:
        raise ValueError("No spatial dimensions found (lat/lon, j/i o x/y).")
    return spatial_dims

def process_model(model_name, base_dir):
    """Procesa un modelo específico cargando y ajustando sus datos."""
    model_path = os.path.join(base_dir, model_name)
    
    # Filtrar archivos válidos: que terminen en .nc, no empiecen con '.' y sean archivos regulares
    model_files = [
        os.path.join(model_path, f)
        for f in os.listdir(model_path)
        if f.endswith(".nc") and not f.startswith(".") and os.path.isfile(os.path.join(model_path, f))
    ]
    
    if not model_files:
        print(f"No valid nc files found in {model_path}")
        return None

    # Cargar múltiples archivos del modelo
    model_ds = xr.open_mfdataset(model_files, concat_dim='time', combine='nested')

    # Corregir coordenadas de tiempo
    model_ds = handle_time_coordinates(model_ds)

    # Ajustar longitudes
    model_ds = adjust_longitudes_and_latitudes(model_ds)

    # Detectar dimensiones espaciales
    spatial_dims = detect_spatial_dimensions(model_ds)

    print(f"{model_name}: Dataset processed for spatial dimensions {spatial_dims}.")

    return model_ds


def process_all_models(selected_models, base_dir):
    """Procesa todos los modelos en la lista seleccionada."""
    processed_models = {}

    for model_name in selected_models:
        try:
            print(f"Process model: {model_name}")
            model_ds = process_model(model_name, base_dir)
            if model_ds is not None:
                processed_models[model_name] = model_ds
        except Exception as e:
            print(f"Error model process for {model_name}: {e}")

    return processed_models


def process_model_variables(model_name, base_dir):
    """
    Procesa todas las variables de un modelo CMIP6, aplica una función de regrilla
    y combina las variables en un único dataset.
    """
    model_path = os.path.join(base_dir, model_name)
    variables = [var for var in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, var))]
    
    datasets = []  # Lista para almacenar los datasets de cada variable
    
    for var in variables:
        var_path = os.path.join(model_path, var)
        
        # Filtrar archivos válidos para NetCDF
        var_files = [
            os.path.join(var_path, f)
            for f in os.listdir(var_path)
            if f.endswith(".nc") and not f.startswith(".") and os.path.isfile(os.path.join(var_path, f))
        ]
        
        if not var_files:
            print(f"No files found for '{var}' in model '{model_name}'")
            continue
        
        # Abrir múltiples archivos como un solo dataset
        try:
            ds = xr.open_mfdataset(var_files, concat_dim='time', combine='nested')
        except Exception as e:
            print(f"Error to open file '{var}' en '{model_name}': {e}")
            continue 
        ds = handle_time_coordinates2(ds)
        #ds = adjust_longitudes_and_latitudes(ds)
        spatial_dims = detect_spatial_dimensions(ds)
        
        ds=ds.sel(time=slice('1970','2014'))
        
        # Aplicar función de regrilla
        try:
            regridded_ds = regrid_models_simple(ds)
        except Exception as e:
            print(f"error regrid for '{var}' in '{model_name}': {e}")
            continue
        
        # Añadir al dataset final asignando el nombre de la variable
        try:
            #regridded_ds = regridded_ds.rename({list(regridded_ds.data_vars)[0]: var})
            datasets.append(regridded_ds)
        except Exception as e:
            print(f"error to rename '{var}' in '{model_name}': {e}")
            continue
    
    # Combinar todas las variables en un único dataset
    if datasets:
        try:
            combined_ds = xr.merge(datasets,compat='override')
            return combined_ds
        except Exception as e:
            print(f"error in combining dataset for '{model_name}': {e}")
            return None
    else:
        print(f"error not variables processed for '{model_name}'")
        return None


def process_models_forcing(base_dir):
    """Procesa todos los modelos en el directorio base y aplica una función de regrilla."""
    model_names = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
    output_dir = "/Volumes/Thalassa/CMIP6_forcings_RG_360"
    os.makedirs(output_dir, exist_ok=True)
    
    expected_variables = ['hfds', 'sos', 'tauv', 'evspsbl', 'tauu', 'tos','pr']
    
    all_regridded_data = {}

    for model_name in model_names:
        output_file = os.path.join(output_dir, f"{model_name}_atm_forcing.nc")
        
        if os.path.exists(output_file):
        
            existing_ds = xr.open_dataset(output_file)
        
            missing_vars = [var for var in expected_variables if var not in existing_ds]
            existing_ds.close()
            print(missing_vars)
            if not missing_vars:
                print(f"Archivo ya existe para '{model_name}', saltando procesamiento.")
                continue
            else:
                print(f"Procesando modelo: {model_name}")
                try:
                    combined_ds=process_model_variables(model_name, base_dir)
                    combined_ds.to_netcdf(output_file)
                except Exception as e:
                    print(f"error in '{model_name}': {e}")
                    continue
        else:
            try:
                combined_ds=process_model_variables(model_name, base_dir)
                combined_ds.to_netcdf(output_file)
            except Exception as e:
                print(f"error in '{model_name}': {e}")
                continue

        # Verificar si el archivo ya existe

        
        #all_regridded_data[model_name] = process_model_variables(model_name, base_dir)
        

        
    return model_names#all_regridded_data


def handle_time_coordinates2(model_ds):
    """
    Convierte las coordenadas de tiempo en el dataset de cftime a numpy.datetime64
    usando strings como intermediarios.
    """
    try:
        # Convertir coordenadas de tiempo a strings y luego a numpy.datetime64
        if isinstance(model_ds.time.values[0], cftime.datetime):
            model_ds['time'] = np.array([str(t) for t in model_ds['time'].values], dtype="datetime64[ns]")
        
        # Si las fechas son de tipo cftime.Datetime360Day, convertirlas directamente
        if isinstance(model_ds.indexes['time'][0], cftime.Datetime360Day):
            model_ds = model_ds.assign_coords(time=model_ds.indexes['time'].to_datetimeindex())
            
        if len(np.unique(model_ds['time'])) < len(model_ds['time']):
            _, index = np.unique(model_ds['time'], return_index=True)
            unique_mask = np.isin(np.arange(len(model_ds['time'])), index)
            model_ds = model_ds.isel(time=unique_mask)
    
    except Exception as e:
        print(f"Advertencia: No se pudo convertir las fechas a numpy.datetime64. Error: {e}")
        raise e  # Opcional, dependiendo de si quieres continuar o detener el flujo.
        
    model_ds = model_ds.sortby('time')
    
    return model_ds

