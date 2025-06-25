"""
Format Converter: converts input data formats into SedtrailsData format.

This module reads various input data formats (e.g., NetCDF files from different
hydrodynamic models) and converts them into the SedtrailsData structure for
use in the SedTRAILS particle tracking system.
"""

import numpy as np
import xarray as xr
import xugrid as xu
from enum import Enum
from typing import Union, Dict
from pathlib import Path
from datetime import datetime
from sedtrails.transport_converter.sedtrails_data import SedtrailsData


class InputType(Enum):
    """Enumeration of supported input data types."""

    NETCDF_DFM = 'netcdf_dfm'  # Delft3D Flexible Mesh NetCDF
    TRIM_D3D4 = 'trim_d3d4'  # Delft3D4 TRIM format (placeholder, not implemented yet)
    # Add more input types as needed


class FormatConverter:
    """
    A class to convert various input data formats to the SedtrailsData format.

    This class provides methods to read data from different file formats and
    convert them to the SedtrailsData format for use in the SedTRAILS particle
    tracking system.
    """

    def __init__(
        self,
        input_file: Union[str, Path],
        input_type: Union[str, InputType] = InputType.NETCDF_DFM,
        reference_date: Union[str, np.datetime64, datetime] = '1970-01-01',
    ):
        """
        Initialize the FormatConverter.

        Parameters:
        -----------
        input_file : str or Path
            Path to the input file
        input_type : str or InputType, optional
            Type of input data, by default InputType.NETCDF_DFM
        reference_date : str, np.datetime64, or datetime, optional
            Reference date for time values, by default "1970-01-01" (Unix epoch)
        """
        self.input_file = Path(input_file)

        # Set input type
        if isinstance(input_type, str):
            try:
                self.input_type = InputType(input_type.lower())
            except ValueError:
                raise ValueError(
                    f'Invalid input type: {input_type}. Must be one of {[t.value for t in InputType]}'
                ) from ValueError
        else:
            self.input_type = input_type

        # Set reference date
        if isinstance(reference_date, str):
            self.reference_date = np.datetime64(reference_date)
        elif isinstance(reference_date, datetime):
            self.reference_date = np.datetime64(reference_date)
        else:
            self.reference_date = reference_date

        self.input_data = None

    def read_data(self) -> None:
        """
        Read the input data based on the input type.
        """
        if not self.input_file.exists():
            raise FileNotFoundError(f'Input file not found: {self.input_file}')

        if self.input_type == InputType.NETCDF_DFM:
            self._read_netcdf_dfm()
        elif self.input_type == InputType.TRIM_D3D4:
            # Placeholder for future implementation
            raise NotImplementedError('TRIM_D3D4 format not implemented yet')
        else:
            raise NotImplementedError(f'Input type not implemented: {self.input_type}')

    def _read_netcdf_dfm(self) -> None:
        """
        Read a Delft3D Flexible Mesh NetCDF file using xugrid.
        """
        try:
            # First try using xugrid's open_dataset which handles UGRID conventions
            self.input_data = xu.open_dataset(self.input_file, decode_timedelta=True)
        except Exception as e:
            print(f'Could not open file with xugrid: {e}')
            # Fallback to regular xarray
            try:
                self.input_data = xr.open_dataset(self.input_file, decode_timedelta=True)
            except Exception as e:
                raise IOError(f'Failed to open NetCDF file: {e}') from e

        print(f'Successfully loaded {self.input_file}')
        print(f'Variables in dataset: {list(self.input_data.data_vars)}')

    def get_time_info(self) -> Dict:
        """
        Get time information from the dataset.

        Returns:
        --------
        Dict
            Dictionary containing time values, start time, end time,
            and time in seconds since reference date
        """
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call read_data() first.')

        # Get the time variable
        time_var = self.input_data['time']
        time_values = time_var.values
        time_start = time_values[0]
        time_end = time_values[-1]

        # Get original time units and calendar from the attributes
        orig_units = getattr(time_var, 'units', None)
        orig_calendar = getattr(time_var, 'calendar', 'standard')

        # Convert time values to seconds since reference_date
        seconds_since_ref = np.array([float((t - self.reference_date) / np.timedelta64(1, 's')) for t in time_values])

        return {
            'time_values': time_values,
            'time_start': time_start,
            'time_end': time_end,
            'original_units': orig_units,
            'original_calendar': orig_calendar,
            'seconds_since_reference': seconds_since_ref,
            'reference_date': self.reference_date,
            'num_times': len(time_values),
        }

    def _map_dfm_variables(self) -> Dict:
        """
        Map Delft3D Flexible Mesh variables to SedtrailsData structure.

        This function processes all time steps at once.

        Returns:
        --------
        Dict
            Dictionary with mapped variables
        """
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call read_data() first.')

        # Get time information
        time_info = self.get_time_info()
        num_times = time_info['num_times']

        # Variable mapping for DFM files
        variable_map = {
            'x': 'net_xcc',  # X-coordinates
            'y': 'net_ycc',  # Y-coordinates
            'bed_level': 'bedlevel',  # Bed level
            'water_depth': 'waterdepth',  # Water depth
            'flow_velocity_x': 'sea_water_x_velocity',  # X-component of flow velocity
            'flow_velocity_y': 'sea_water_y_velocity',  # Y-component of flow velocity
            'mean_bed_shear_stress': 'mean_bss_magnitude',  # Mean bed shear stress
            'max_bed_shear_stress': 'max_bss_magnitude',  # Max bed shear stress
            'bed_load_transport_x': 'bedload_x_comp',  # X-component of bed load sediment transport
            'bed_load_transport_y': 'bedload_y_comp',  # Y-component of bed load sediment transport
            'suspended_transport_x': 'susload_x_comp',  # X-component of suspended sediment transport
            'suspended_transport_y': 'susload_y_comp',  # Y-component of suspended sediment transport
            'sediment_concentration': 'suspended_sed_conc',  # Suspended sediment concentration
        }

        # Extract data from dataset
        data = {}

        # First, get spatial coordinates (typically not time-dependent)
        for key in ['x', 'y']:
            var_name = variable_map[key]
            if var_name in self.input_data:
                data[key] = self.input_data[var_name].values
            else:
                raise KeyError(f"Required variable '{var_name}' not found in dataset")

        # Determine the spatial grid dimensions
        grid_shape = data['x'].shape

        # Get bed level (typically not time-dependent)
        var_name = variable_map['bed_level']
        if var_name in self.input_data:
            bed_level_var = self.input_data[var_name]
            if 'time' in bed_level_var.dims:
                # If bed level has a time dimension, take the first time step
                data['bed_level'] = bed_level_var.isel(time=0).values
            else:
                data['bed_level'] = bed_level_var.values
        else:
            # Default to zeros if not found
            data['bed_level'] = np.zeros(grid_shape)
            print(f"Warning: Variable '{var_name}' not found, using zeros")

        # Extract time-dependent variables
        time_dependent_vars = [
            'water_depth',
            'mean_bed_shear_stress',
            'max_bed_shear_stress',
            'sediment_concentration',
            'flow_velocity_x',
            'flow_velocity_y',
            'bed_load_transport_x',
            'bed_load_transport_y',
            'suspended_transport_x',
            'suspended_transport_y',
        ]

        for key in time_dependent_vars:
            var_name = variable_map[key]
            if var_name in self.input_data:
                var = self.input_data[var_name]

                # Check if variable has time dimension
                if 'time' in var.dims:
                    # Check if variable has layer dimension
                    if 'layer' in var.dims:
                        # For variables with time and layer, select layer 0
                        data[key] = var.isel(layer=0).values
                    else:
                        # For variables with time but no layer
                        data[key] = var.values
                else:
                    # For variables without time dimension, broadcast to all time steps
                    data[key] = np.broadcast_to(var.values, (num_times, *var.shape))
            else:
                # Default to zeros if not found
                data[key] = np.zeros((num_times, *grid_shape))
                print(f"Warning: Variable '{var_name}' not found, using zeros")

        return data

    def convert_to_sedtrails_data(self) -> SedtrailsData:
        """
        Convert the loaded dataset to SedtrailsData format for all time steps.

        Returns:
        --------
        SedtrailsData
            Data in SedtrailsData format with time as the first dimension for
            time-dependent variables, with time in seconds since reference_date
        """
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call read_data() first.')

        # Get time information
        time_info = self.get_time_info()
        seconds_since_ref = time_info['seconds_since_reference']

        # Get mapped variables based on input type
        if self.input_type == InputType.NETCDF_DFM:
            data = self._map_dfm_variables()
        else:
            raise NotImplementedError(f'Conversion not implemented for input type: {self.input_type}')

        # Calculate magnitudes for vector quantities
        # Flow velocity magnitude
        depth_avg_velocity_magnitude = np.sqrt(data['flow_velocity_x'] ** 2 + data['flow_velocity_y'] ** 2)

        # Bed load magnitude
        bed_load_magnitude = np.sqrt(data['bed_load_transport_x'] ** 2 + data['bed_load_transport_y'] ** 2)

        # Suspended sediment magnitude
        suspended_transport_magnitude = np.sqrt(data['suspended_transport_x'] ** 2 + data['suspended_transport_y'] ** 2)

        # Create dictionaries for vector quantities
        depth_avg_flow_velocity = {
            'x': data['flow_velocity_x'],
            'y': data['flow_velocity_y'],
            'magnitude': depth_avg_velocity_magnitude,
        }

        bed_load_transport = {
            'x': data['bed_load_transport_x'],
            'y': data['bed_load_transport_y'],
            'magnitude': bed_load_magnitude,
        }

        suspended_transport = {
            'x': data['suspended_transport_x'],
            'y': data['suspended_transport_y'],
            'magnitude': suspended_transport_magnitude,
        }

        # Create nonlinear wave velocity dictionary with zeros
        # Using the same shape as other vector quantities
        nonlinear_wave_velocity = {
            'x': np.zeros_like(data['flow_velocity_x']),
            'y': np.zeros_like(data['flow_velocity_y']),
            'magnitude': np.zeros_like(depth_avg_velocity_magnitude),
        }

        # Create SedtrailsData object
        sedtrails_data = SedtrailsData(
            times=seconds_since_ref,
            reference_date=self.reference_date,
            x=data['x'],
            y=data['y'],
            bed_level=data['bed_level'],
            depth_avg_flow_velocity=depth_avg_flow_velocity,
            fractions=1,  # Default to 1 fraction
            bed_load_transport=bed_load_transport,
            suspended_transport=suspended_transport,
            water_depth=data['water_depth'],
            mean_bed_shear_stress=data['mean_bed_shear_stress'],
            max_bed_shear_stress=data['max_bed_shear_stress'],
            sediment_concentration=data['sediment_concentration'],
            nonlinear_wave_velocity=nonlinear_wave_velocity,
        )

        return sedtrails_data


# Note: The example code has been moved to examples/format_converter_example.py
if __name__ == '__main__':
    print('Please see the examples directory for usage examples.')
