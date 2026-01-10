"""
Sensor Unit Conversion Module for SmartFallMM Dataset.

Handles the different data formats across sensor modalities:
- Meta sensors (wrist/hip): Accelerometer in g-units, Gyroscope in deg/s
- Android sensors (watch/phone): Accelerometer in m/s², Gyroscope in deg/s

This module provides:
1. Automatic sensor type detection from file path
2. Unit conversion functions (g→m/s², deg/s→rad/s)
3. Integration with the loader.py preprocessing pipeline

Usage:
    from utils.sensor_units import SensorUnitConverter

    converter = SensorUnitConverter()
    acc_data = converter.convert_accelerometer(data, sensor='meta_wrist')
    gyro_data = converter.convert_gyroscope(data, sensor='meta_wrist')

Author: SmartFall Research Group
Date: December 2024
"""

import numpy as np
from typing import Union, Optional, Tuple
from enum import Enum


class SensorType(Enum):
    """Enumeration of sensor types in SmartFallMM dataset."""
    WATCH = "watch"
    PHONE = "phone"
    META_WRIST = "meta_wrist"
    META_HIP = "meta_hip"


# Sensor configuration mapping
SENSOR_CONFIG = {
    # Android sensors (watch/phone)
    SensorType.WATCH: {
        'acc_unit': 'm/s^2',      # Already in m/s²
        'gyro_unit': 'deg/s',     # Needs conversion to rad/s
        'nominal_fs': 32.0,       # Variable sampling
        'recommended_filter_fs': 30.0
    },
    SensorType.PHONE: {
        'acc_unit': 'm/s^2',      # Already in m/s²
        'gyro_unit': 'deg/s',     # Needs conversion to rad/s
        'nominal_fs': 32.0,       # Variable sampling
        'recommended_filter_fs': 30.0
    },
    # Meta sensors (wrist/hip)
    SensorType.META_WRIST: {
        'acc_unit': 'g',          # Needs conversion to m/s²
        'gyro_unit': 'deg/s',     # Needs conversion to rad/s
        'nominal_fs': 50.0,       # Fixed sampling
        'recommended_filter_fs': 50.0
    },
    SensorType.META_HIP: {
        'acc_unit': 'g',          # Needs conversion to m/s²
        'gyro_unit': 'deg/s',     # Needs conversion to rad/s
        'nominal_fs': 50.0,       # Fixed sampling
        'recommended_filter_fs': 50.0
    }
}

# Physical constants
GRAVITY_MPS2 = 9.81  # m/s²
DEG_TO_RAD = np.pi / 180.0


def detect_sensor_type(file_path: str) -> Optional[SensorType]:
    """
    Detect sensor type from file path.

    Args:
        file_path: Path to sensor data file

    Returns:
        SensorType enum or None if unrecognized
    """
    path_lower = file_path.lower()

    if 'meta_wrist' in path_lower:
        return SensorType.META_WRIST
    elif 'meta_hip' in path_lower:
        return SensorType.META_HIP
    elif 'watch' in path_lower:
        return SensorType.WATCH
    elif 'phone' in path_lower:
        return SensorType.PHONE
    else:
        return None


def convert_acc_g_to_mps2(data: np.ndarray) -> np.ndarray:
    """
    Convert accelerometer data from g-units to m/s².

    Args:
        data: (N, 3) accelerometer data in g-units

    Returns:
        (N, 3) accelerometer data in m/s²
    """
    return data * GRAVITY_MPS2


def convert_gyro_deg_to_rad(data: np.ndarray) -> np.ndarray:
    """
    Convert gyroscope data from deg/s to rad/s.

    Args:
        data: (N, 3) gyroscope data in deg/s

    Returns:
        (N, 3) gyroscope data in rad/s
    """
    return data * DEG_TO_RAD


def is_acc_in_g_units(data: np.ndarray, threshold: float = 2.0) -> bool:
    """
    Heuristically detect if accelerometer data is in g-units.

    At rest, |a| should be ~1g or ~9.81 m/s².

    Args:
        data: (N, 3) accelerometer data
        threshold: If median magnitude < threshold, assume g-units

    Returns:
        True if data appears to be in g-units
    """
    magnitudes = np.linalg.norm(data, axis=1)
    median_mag = np.median(magnitudes)

    # If median is ~1, it's g-units; if ~9.8, it's m/s²
    return median_mag < threshold


def is_gyro_in_deg(data: np.ndarray, threshold: float = 10.0) -> bool:
    """
    Heuristically detect if gyroscope data is in deg/s.

    Normal human motion rarely exceeds 10 rad/s (~570 deg/s).
    Falls can produce up to ~2-3 rad/s (~115-170 deg/s).

    Args:
        data: (N, 3) gyroscope data
        threshold: If max value > threshold, assume deg/s

    Returns:
        True if data appears to be in deg/s
    """
    max_val = np.abs(data).max()
    return max_val > threshold


class SensorUnitConverter:
    """
    Unified sensor unit conversion with automatic detection.

    Handles both explicit sensor specification and auto-detection.

    Usage:
        converter = SensorUnitConverter(auto_detect=True)

        # Option 1: Explicit sensor type
        acc_data = converter.convert_accelerometer(data, sensor='meta_wrist')

        # Option 2: Auto-detection from file path
        acc_data = converter.convert_accelerometer(data, file_path='/path/to/meta_wrist/file.csv')

        # Option 3: Auto-detection from data statistics
        acc_data = converter.convert_accelerometer(data)  # Heuristic detection
    """

    def __init__(self, auto_detect: bool = True, verbose: bool = False):
        """
        Initialize converter.

        Args:
            auto_detect: If True, automatically detect units from data statistics
            verbose: If True, print conversion messages
        """
        self.auto_detect = auto_detect
        self.verbose = verbose
        self._conversion_count = {'acc': 0, 'gyro': 0}

    def _get_sensor_type(self,
                        sensor: Optional[str] = None,
                        file_path: Optional[str] = None) -> Optional[SensorType]:
        """Get sensor type from explicit name or file path."""
        if sensor is not None:
            sensor_lower = sensor.lower()
            for st in SensorType:
                if st.value == sensor_lower:
                    return st
            return None

        if file_path is not None:
            return detect_sensor_type(file_path)

        return None

    def convert_accelerometer(self,
                             data: np.ndarray,
                             sensor: Optional[str] = None,
                             file_path: Optional[str] = None,
                             force_convert: bool = False) -> np.ndarray:
        """
        Convert accelerometer data to m/s² if needed.

        Args:
            data: (N, 3) accelerometer data
            sensor: Sensor name ('watch', 'phone', 'meta_wrist', 'meta_hip')
            file_path: Path to data file (for auto-detection)
            force_convert: Force g→m/s² conversion regardless of detection

        Returns:
            (N, 3) accelerometer data in m/s²
        """
        sensor_type = self._get_sensor_type(sensor, file_path)

        # Determine if conversion is needed
        needs_conversion = False

        if force_convert:
            needs_conversion = True
        elif sensor_type is not None:
            config = SENSOR_CONFIG[sensor_type]
            needs_conversion = config['acc_unit'] == 'g'
        elif self.auto_detect:
            needs_conversion = is_acc_in_g_units(data)

        if needs_conversion:
            if self.verbose:
                print(f"[SensorUnitConverter] Converting accelerometer from g-units to m/s²")
            self._conversion_count['acc'] += 1
            return convert_acc_g_to_mps2(data)

        return data

    def convert_gyroscope(self,
                         data: np.ndarray,
                         sensor: Optional[str] = None,
                         file_path: Optional[str] = None,
                         force_convert: bool = False) -> np.ndarray:
        """
        Convert gyroscope data to rad/s if needed.

        Args:
            data: (N, 3) gyroscope data
            sensor: Sensor name ('watch', 'phone', 'meta_wrist', 'meta_hip')
            file_path: Path to data file (for auto-detection)
            force_convert: Force deg→rad conversion regardless of detection

        Returns:
            (N, 3) gyroscope data in rad/s
        """
        sensor_type = self._get_sensor_type(sensor, file_path)

        # Determine if conversion is needed
        needs_conversion = False

        if force_convert:
            needs_conversion = True
        elif sensor_type is not None:
            config = SENSOR_CONFIG[sensor_type]
            needs_conversion = config['gyro_unit'] == 'deg/s'
        elif self.auto_detect:
            needs_conversion = is_gyro_in_deg(data)

        if needs_conversion:
            if self.verbose:
                print(f"[SensorUnitConverter] Converting gyroscope from deg/s to rad/s")
            self._conversion_count['gyro'] += 1
            return convert_gyro_deg_to_rad(data)

        return data

    def get_recommended_kalman_params(self, sensor: str) -> dict:
        """
        Get recommended Kalman filter parameters for a sensor type.

        Args:
            sensor: Sensor name ('watch', 'phone', 'meta_wrist', 'meta_hip')

        Returns:
            Dictionary of recommended Kalman parameters
        """
        sensor_type = self._get_sensor_type(sensor)

        if sensor_type is None:
            # Default parameters
            return {
                'filter_fs': 30.0,
                'kalman_Q_orientation': 0.005,
                'kalman_Q_rate': 0.01,
                'kalman_R_acc': 0.05,
                'kalman_R_gyro': 0.1
            }

        config = SENSOR_CONFIG[sensor_type]

        if sensor_type in [SensorType.WATCH, SensorType.PHONE]:
            # Android sensors: current optimized params
            return {
                'filter_fs': config['recommended_filter_fs'],
                'kalman_Q_orientation': 0.005,
                'kalman_Q_rate': 0.01,
                'kalman_R_acc': 0.05,
                'kalman_R_gyro': 0.1
            }
        else:
            # Meta sensors: adjusted for higher fidelity
            return {
                'filter_fs': config['recommended_filter_fs'],
                'kalman_Q_orientation': 0.01,   # Higher due to faster sampling
                'kalman_Q_rate': 0.02,
                'kalman_R_acc': 0.02,           # Lower - trust Meta acc more
                'kalman_R_gyro': 0.15           # Higher - more motion captured
            }

    def get_conversion_stats(self) -> dict:
        """Return count of conversions performed."""
        return self._conversion_count.copy()


# Convenience functions for integration with loader.py

def convert_sensor_data(modality: str,
                       data: np.ndarray,
                       sensor: str,
                       enable_auto_conversion: bool = True) -> np.ndarray:
    """
    Convenience function for loader.py integration.

    Args:
        modality: 'accelerometer' or 'gyroscope'
        data: (N, 3) sensor data
        sensor: Sensor name ('watch', 'phone', 'meta_wrist', 'meta_hip')
        enable_auto_conversion: Enable automatic unit conversion

    Returns:
        (N, 3) data in standard units (m/s² for acc, rad/s for gyro)
    """
    if not enable_auto_conversion:
        return data

    converter = SensorUnitConverter(auto_detect=True)

    if modality == 'accelerometer':
        return converter.convert_accelerometer(data, sensor=sensor)
    elif modality == 'gyroscope':
        return converter.convert_gyroscope(data, sensor=sensor)
    else:
        return data


def get_sensor_config(sensor: str) -> dict:
    """
    Get sensor configuration including units and sampling rate.

    Args:
        sensor: Sensor name

    Returns:
        Configuration dictionary
    """
    sensor_lower = sensor.lower()
    for st in SensorType:
        if st.value == sensor_lower:
            return SENSOR_CONFIG[st].copy()
    return {}
