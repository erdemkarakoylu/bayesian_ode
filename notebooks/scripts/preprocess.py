import pandas as pd
import xarray as xr


def convert_yrday(yd, year):
    """Convert 'yrday_gmt' to datetime objects, now with the known year"""
    day_of_year = int(yd)
    fractional_day = yd - day_of_year
    start_of_year = pd.to_datetime(f'{year}-01-01')
    time_delta = pd.to_timedelta(fractional_day * 24, unit='h')
    return start_of_year + pd.Timedelta(days=day_of_year - 1) + time_delta


def process_adcp_bathy_time(ds: xr.Dataset):
    year = 1996
    times = pd.Series(ds['yrday_gmt'].values).apply(lambda x: convert_yrday(x, year)).values
    ds['time'] = ('unlimited', times)

def set_bathy_coords_dims(ds: xr.Dataset):
    ds = ds.set_coords(['time', 'lat', 'lon'])
    ds = ds.rename(dict(unlimited='record'))
    return ds


def process_nut_phyto(ds):
    """Process the nutrient and phytoplankton dataset.

    This function orchestrates the steps to add a proper datetime coordinate,
    set relevant variables as coordinates, rename the main dimension, and
    assign units to the data variables.

    Parameters
    ----------
    ds : xarray.Dataset
        The input xarray Dataset containing nutrient and phytoplankton data.
        It is expected to have 'year', 'month', 'day', 'yrday_local', 'lat',
        'lon', 'depth', 'station_std', 'chl_a', 'phaeo', 'press', 'temp',
        'sal', 'flvolt', 'NH4', 'NO3_NO2', 'PO4', and 'SiOH_4' variables.

    Returns
    -------
    xarray.Dataset
        The processed xarray Dataset with a 'time' coordinate, 'lat', 'lon',
        'depth', and 'station_std' set as coordinates, the main dimension
        renamed to 'observation', and units added as attributes to the
        data variables.
    """
    ds = get_date_time(ds)
    ds = set_nut_phyto_coords(ds)
    set_unit_variables(ds)
    return ds


def get_date_time(ds):
    """Create a datetime coordinate from year, month, day, and yrday_local.

    This function converts the 'year', 'month', and 'day' variables into a base
    datetime object (midnight of that day) and then uses the fractional part
    of 'yrday_local' to add the time of day (hours, minutes, seconds) to
    create a 'time' coordinate in the Dataset. The original 'year', 'month',
    'day', 'yrday_local', and the original ambiguous 'time' variables are
    dropped from the Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The input xarray Dataset containing 'year', 'month', 'day', and
        'yrday_local' variables.

    Returns
    -------
    None
        The input Dataset `ds` is modified in place by adding a 'time'
        coordinate and dropping the original time-related variables.

    array(['1997-01-14T12:00:00.000000000'], dtype='datetime64[ns]')
    """
    # Convert year, month, day to integers
    year_int = ds['year'].astype(int)
    month_int = ds['month'].astype(int)
    day_int = ds['day'].astype(int)

    # Create a base datetime object (midnight of that day)
    base_times = pd.to_datetime(dict(year=year_int, month=month_int, day=day_int))
    # Extract the fractional part of yrday_local (representing the time of day)
    fractional_day = ds['yrday_local'] - ds['yrday_local'].astype(int)
    # Convert the fractional day to total seconds
    seconds_in_day = fractional_day * 24 * 3600
    # Add the seconds to the base datetime to get the time of day
    final_times = base_times + pd.to_timedelta(seconds_in_day, unit='s')
    # Add the datetime coordinate to the Dataset
    ds['time'] = ('unlimited', final_times)
    # Drop the original time components (including the ambiguous 'time' variable)
    ds = ds.drop_vars(['year', 'month', 'day', 'yrday_local'])
    return ds


def set_nut_phyto_coords(ds):
    """Set standard coordinates for the nutrient and phytoplankton dataset.

    This function sets 'time', 'lat', 'lon', 'depth', and 'station_std' as
    coordinates of the input xarray Dataset and renames the 'unlimited'
    dimension to 'observation'.

    Parameters
    ----------
    ds : xarray.Dataset
        The input xarray Dataset. It is expected to have 'time', 'lat', 'lon',
        'depth', and 'station_std' as data variables and an 'unlimited'
        dimension.

    Returns
    -------
    None
        The input Dataset `ds` is modified in place by setting coordinates
        and renaming the dimension.

    Coordinates:
      * time         (record) datetime64[ns] 2000-01-01
      * lat          (record) float64 1.0
      * lon          (record) float64 2.0
      * depth        (record) float64 10.0
      * station_std  (record) <U1 'A'
    Dimensions without coordinates: record
    """
    # Set coordinates
    ds = ds.set_coords(['time', 'lat', 'lon', 'depth', 'station_std'])

    # Rename the 'unlimited' dimension
    ds = ds.rename({'unlimited': 'observation'})
    return ds

def set_unit_variables(ds):
    """Set standard units as attributes for key variables.

    This function adds 'units' as attributes to the 'chl_a', 'phaeo', 'depth',
    'lat', 'lon', 'NH4', 'NO3_NO2', 'PO4', 'press', 'sal', 'SiOH_4', and
    'temp' variables in the input xarray Dataset based on the provided
    data description.

    Parameters
    ----------
    ds : xarray.Dataset
        The input xarray Dataset containing the aforementioned data variables.

    Returns
    -------
    None
        The input Dataset `ds` is modified in place by adding 'units'
        attributes to the specified data variables.

    """
    # Add units as attributes (as done previously)
    ds['chl_a'].attrs['units'] = 'mg/m3'
    ds['phaeo'].attrs['units'] = 'mg/m3'
    ds['depth'].attrs['units'] = 'meters'
    ds['lat'].attrs['units'] = 'degrees_north'
    ds['lon'].attrs['units'] = 'degrees_east'
    ds['NH4'].attrs['units'] = 'µM'
    ds['NO3_NO2'].attrs['units'] = 'µM'
    ds['PO4'].attrs['units'] = 'µM'
    ds['press'].attrs['units'] = 'decibars'
    ds['sal'].attrs['units'] = 'practical salinity units'
    ds['SiOH_4'].attrs['units'] = 'µM'
    ds['temp'].attrs['units'] = 'Degrees Centigrade'
    ds['flvolt'].attrs['units'] = 'Volts'
    return ds
