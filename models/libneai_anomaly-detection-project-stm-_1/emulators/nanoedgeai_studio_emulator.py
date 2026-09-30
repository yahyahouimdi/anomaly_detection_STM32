#!/usr/bin/env python3
"""
NanoEdgeAI Studio Emulator - Python Wrapper for libneai Library

Copyright (c) 2025 STMicroelectronics
All rights reserved.

This script provides a Python interface to interact with NanoEdgeAI libraries
compiled by NanoEdgeAI Studio. It supports all model types: anomaly detection,
classification, outlier detection, and extrapolation.

DESCRIPTION:
    The NanoEdgeAI Studio Emulator allows you to:
    - Load and initialize NanoEdgeAI compiled libraries (libneai.dll/.so)
    - Train models using the learn() function (for anomaly detection)
    - Run inference using the detect() function
    - Process CSV data files for batch operations
    - Get structured results with status codes and predictions

USAGE:
    Command Line:
        # Learn from data
        python nanoedgeai_studio_emulator.py --lib path/to/libneai.so --learn data.csv

        # Detect/classify data
        python nanoedgeai_studio_emulator.py --lib path/to/libneai.so --detect data.csv

        # Both learn and detect
        python nanoedgeai_studio_emulator.py --lib path/to/libneai.so --learn train.csv --detect test.csv

    Python API:
        from nanoedgeai_studio_emulator import NanoEdgeAIEmulator, read_csv_data, NeaiState

        # Initialize emulator with library
        with NanoEdgeAIEmulator('path/to/libneai.so') as emulator:
            print(f"Model type: {emulator.model_type}")

            # For anomaly detection: learn phase
            if emulator.model_type == 'anomaly_detection':
                learning_data = read_csv_data('learning_data.csv')
                for signal in learning_data:
                    status = emulator.learn(signal)
                    if status == NeaiState.NEAI_LEARNING_DONE:
                        print("Learning complete!")
                        break

            # Detection/inference phase
            test_data = read_csv_data('test_data.csv')
            for signal in test_data:
                status, result = emulator.detect(signal)
                if status == NeaiState.NEAI_OK:
                    print(f"Result: {result}")

CSV FORMAT:
    CSV files should contain one signal per line, with values separated by
    spaces, commas, or tabs (auto-detected). Each line represents one complete
    signal to be processed.

    Example:
        1.23 4.56 7.89 ...
        2.34 5.67 8.90 ...
        3.45 6.78 9.01 ...

REQUIREMENTS:
    - Python 3.7 or higher
    - ctypes (standard library)
    - A compiled NanoEdgeAI library (libneai.dll on Windows, libneai.so on Linux)

For more information about NanoEdgeAI Studio, visit:
https://stm32ai.st.com/nanoedge-ai/
"""

import ctypes
import dataclasses
import math
import platform
import argparse
import csv
from pathlib import Path
from typing import Optional, List, Dict, Union
from enum import IntEnum


class NeaiState(IntEnum):
    """
    NanoEdgeAI Library Return States

    These status codes are returned by the NanoEdgeAI library functions to indicate
    the result of operations (initialization, learning, detection).

    Attributes:
        NEAI_OK: Operation completed successfully
        NEAI_ERROR: Generic error occurred
        NEAI_NOT_INITIALIZED: Library not initialized (call init() first)
        NEAI_INVALID_PARAM: Invalid parameters passed to function
        NEAI_NOT_SUPPORTED: Operation not supported by this model type
        NEAI_LEARNING_DONE: Learning phase completed (anomaly detection)
        NEAI_LEARNING_IN_PROGRESS: Learning still in progress (anomaly detection)

    Example:
        status = emulator.learn(signal_data)
        if status == NeaiState.NEAI_LEARNING_DONE:
            print("Learning complete, ready for detection")
        elif status == NeaiState.NEAI_LEARNING_IN_PROGRESS:
            print("Continue learning with more data")
    """
    NEAI_OK = 0
    NEAI_ERROR = 1
    NEAI_NOT_INITIALIZED = 2
    NEAI_INVALID_PARAM = 3
    NEAI_NOT_SUPPORTED = 4
    NEAI_LEARNING_DONE = 5
    NEAI_LEARNING_IN_PROGRESS = 6


# Model type definitions with corresponding library function names
# This maps each model type to its specific C function names in the library
MODEL_TYPES: Dict[str, Dict[str, Dict | str]] = {
    'anomaly_detection': {
        'display_name': 'Anomaly Detection',
        'functions': {
            'init': 'neai_anomalydetection_init',
            'learn': 'neai_anomalydetection_learn',
            'detect': 'neai_anomalydetection_detect'
        }
    },
    'classification': {
        'display_name': 'Classification',
        'functions': {
            'init': 'neai_classification_init',
            'detect': 'neai_classification'
        }
    },
    'outlier_detection': {
        'display_name': 'Outlier Detection',
        'functions': {
            'init': 'neai_outlier_init',
            'detect': 'neai_outlier'
        }
    },
    'extrapolation': {
        'display_name': 'Extrapolation',
        'functions': {
            'init': 'neai_extrapolation_init',
            'detect': 'neai_extrapolation'
        }
    }
}

ANOMALY_DETECTION_THRESHOLD = 90  # Minimal similarity score threshold for anomaly detection to be considered normal


@dataclasses.dataclass
class DetectionResult:
    state: NeaiState
    value: Union[int, float]
    class_name: Optional[str] = None
    ad_is_nominal: Optional[bool] = None


class NanoEdgeAIEmulator:
    """
    NanoEdgeAI Library Emulator

    This class provides a Python interface to interact with compiled NanoEdgeAI libraries.
    It automatically detects the model type (anomaly detection, classification, outlier
    detection, or extrapolation) and provides appropriate methods for initialization,
    learning (when available), and inference.

    The class supports context manager protocol for automatic resource cleanup.

    Attributes:
        lib: The loaded ctypes library object
        model_type: Detected model type ('anomaly_detection', 'classification',
                    'outlier_detection', or 'extrapolation')
        class_number: Number of classes (for classification models only)
        class_names: List of class names (for classification models only)

    Example:
        # Basic usage with context manager
        with NanoEdgeAIEmulator('libneai.so') as emulator:
            print(f"Model type: {emulator.model_type}")

            # Learn (anomaly detection only)
            if emulator.model_type == 'anomaly_detection':
                status = emulator.learn(training_signal)

            # Detect/classify
            status, result = emulator.detect(test_signal)
            if status == NeaiState.NEAI_OK:
                print(f"Result: {result}")

            # Get class names (classification only)
            if emulator.model_type == 'classification':
                for i in range(emulator.class_number):
                    print(f"Class {i}: {emulator.get_class_name(i)}")
    """

    def __init__(self, lib_path: Optional[str] = None, ad_use_embedded_knowledge: Optional[bool]= False) -> None:
        """
        Initialize the NanoEdgeAI emulator with the specified library.

        Args:
            lib_path: Path to the libneai library file (.dll on Windows, .so on Linux).
                     If None, searches for 'libneai.dll' or 'libneai.so' in the same
                     directory as this script.
            ad_use_embedded_knowledge: For anomaly detection models, whether to use embedded knowledge
                                      during initialization (default: False). Ignored for other model types.

        Raises:
            FileNotFoundError: If the library file cannot be found
            RuntimeError: If the model type cannot be detected from the library
            OSError: If the library cannot be loaded

        Example:
            # Explicit path
            emulator = NanoEdgeAIEmulator('/path/to/libneai.so')

            # Auto-detect in current directory
            emulator = NanoEdgeAIEmulator()
        """
        # Args
        self._lib_path_arg = lib_path
        self.ad_use_embedded_knowledge = ad_use_embedded_knowledge

        # Library
        self._lib_path = None
        self.lib = None
        self._lib_handle = None

        # Initialize function references
        self._init_func = None
        self._learn_func = None
        self._detect_func = None

        # Metadata
        self.model_type = None
        self.model_display_name = None
        self.class_number = None
        self.input_signal_size = None
        self.axis_number = None
        self.class_names: List[Optional[str]] = []

    def __enter__(self) -> 'NanoEdgeAIEmulator':
        """
        Enter context manager.

        Returns:
            self: The emulator instance
        """
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Exit context manager and clean up resources.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)

        Returns:
            False to propagate any exceptions
        """
        self.close()
        return False

    def open(self):
        """
        Prepare and load the NanoEdgeAI library.

        Raises:
            RuntimeError: If the library is already open
        """
        if self._lib_handle is not None:
            raise RuntimeError('Library is already open')

        # Load the library
        self._find_library()
        self.lib = ctypes.CDLL(str(self._lib_path))
        self._lib_handle = self.lib._handle

        # Detect model type and setup functions
        self._detect_model_type()
        self._setup_functions()

        # Call init function before querying model metadata
        self._init()
        self._detect_metadata()

    def close(self) -> None:
        """
        Unload the library and free resources.

        This method is automatically called when using the context manager,
        but can also be called manually if needed.
        """
        # Clear references and let Python's garbage collector handle cleanup.
        # Manual dlclose() is problematic with ctypes and can cause segfaults.
        self._lib_handle = None
        self.lib = None

    @property
    def data_length(self) -> Optional[int]:
        """
        Get the expected data length for the model.

        Returns:
            int: data length, or None if not available
        """
        return self.input_signal_size * self.axis_number if self.input_signal_size and self.axis_number else None

    def _detect_model_type(self):
        """
        Automatically detect the model type from available library functions.

        Inspects the loaded library to determine which NanoEdgeAI model type it contains
        by checking for the presence of specific function names.

        Raises:
            RuntimeError: If no matching model type can be determined
        """
        for model_type, model_conf in MODEL_TYPES.items():
            if all(hasattr(self.lib, func_name) for func_name in model_conf['functions'].values()):
                self.model_type = model_type
                self.model_display_name = model_conf['display_name']
                return

        raise RuntimeError('Could not detect model type from library functions')

    def _find_library(self):
        """
        Locate the NanoEdgeAI library file based on the current platform.

        Searches for 'libneai.dll' on Windows or 'libneai.so' on Linux/Unix systems
        in the same directory as this script.

        Raises:
            FileNotFoundError: If the library file cannot be found
        """
        if not self._lib_path_arg:
            lib_name = 'libneai.dll' if platform.system() == 'Windows' else 'libneai.so'
            self._lib_path = Path(__file__).parent / lib_name
        else:
            self._lib_path = Path(self._lib_path_arg)

        if not self._lib_path.exists():
            if self._lib_path_arg:
                raise FileNotFoundError(f'Library not found. File not found at specified path: {self._lib_path_arg}')
            else:
                raise FileNotFoundError(f'Library not found. Auto-detected path: {self._lib_path}')

    def _setup_functions(self) -> None:
        """
        Configure function signatures (argtypes and restype) for all library functions.

        This method sets up the correct C function signatures based on the detected
        model type, ensuring proper type conversion between Python and C. Each model
        type may have different function signatures for detection.

        Function signatures:
            - Anomaly Detection: init(uint8 embedded_knowledge), detect(float *data, uint8 *similarity)
            - Classification: detect(float *input, float *output_buffer, int *class_id)
            - Outlier Detection: detect(float *data, uint8 *is_outlier)
            - Extrapolation: detect(float *input, float *output)
        """
        functions = MODEL_TYPES[self.model_type]['functions']

        # Init
        self._init_func = getattr(self.lib, functions['init'])
        self._init_func.restype = ctypes.c_int
        if self.model_type == 'anomaly_detection':
            # Anomaly detection init takes embedded_knowledge parameter
            self._init_func.argtypes = [ctypes.c_uint8]
        else:
            self._init_func.argtypes = []
            self.ad_use_embedded_knowledge = False

        # Learn (if available)
        if 'learn' in functions:
            self._learn_func = getattr(self.lib, functions['learn'])
            self._learn_func.restype = ctypes.c_int
            self._learn_func.argtypes = [ctypes.POINTER(ctypes.c_float)]

        # Detect
        self._detect_func = getattr(self.lib, functions['detect'])
        self._detect_func.restype = ctypes.c_int

        if self.model_type == 'classification':
            # Classification: (float *input, float *output_buffer, int *class_id)
            self._detect_func.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int)
            ]
        elif self.model_type == 'extrapolation':
            # Extrapolation: (float *input, float *output)
            self._detect_func.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float)
            ]
        else:
            # Anomaly/Outlier detection: (float *data, uint8 *similarity/is_outlier)
            self._detect_func.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_uint8)
            ]

    def _detect_metadata(self) -> None:
        """Gather and cache all library metadata once the model is initialized."""
        if hasattr(self.lib, 'neai_get_input_signal_size'):
            get_input_size = getattr(self.lib, 'neai_get_input_signal_size')
            get_input_size.restype = ctypes.c_int
            self.input_signal_size = get_input_size()
        else:
            raise RuntimeError('Input signal size function not found in the library')

        if hasattr(self.lib, 'neai_get_axis_number'):
            get_axis_number = getattr(self.lib, 'neai_get_axis_number')
            get_axis_number.restype = ctypes.c_int
            self.axis_number = get_axis_number()
        else:
            raise RuntimeError('Axis number function not found in the library')

        if self.model_type == 'classification':
            if hasattr(self.lib, 'neai_get_number_of_classes'):
                get_class_num = getattr(self.lib, 'neai_get_number_of_classes')
                get_class_num.restype = ctypes.c_int
                self.class_number = get_class_num()
            else:
                raise RuntimeError('Classification model detected but class number function not found')

            self.class_names = []
            if hasattr(self.lib, 'neai_get_class_name'):
                class_name_func = getattr(self.lib, 'neai_get_class_name')
                class_name_func.restype = ctypes.c_char_p
                class_name_func.argtypes = [ctypes.c_int]
                for i in range(self.class_number):
                    name = class_name_func(i)
                    self.class_names.append(name.decode('utf-8') if name else None)
            else:
                raise RuntimeError('Classification model detected but class name function not found')
        else:
            self.class_number = None
            self.class_names = []


    def _get_data_array(self, data: List[float]) -> ctypes.Array:
        if self.data_length is not None and len(data) != self.data_length:
            raise ValueError(f'Expected signal length {self.data_length}, got {len(data)}')
        return (ctypes.c_float * len(data))(*data)

    def _init(self) -> int:
        """
        Initialize the NanoEdgeAI model.

        This method is automatically called during emulator initialization.

        Returns:
            int: Initialization status (0 for success)
        """
        if self.model_type == 'anomaly_detection':
            embedded_knowledge = ctypes.c_uint8(1 if self.ad_use_embedded_knowledge else 0)
            return self._init_func(embedded_knowledge)
        return self._init_func()

    def learn(self, data: List[float]) -> NeaiState:
        """
        Train the model with a single signal (anomaly detection only).

        This method is only available for anomaly detection models. It trains the
        model to recognize normal behavior patterns. Multiple calls are typically
        needed until NEAI_LEARNING_DONE is returned.

        Args:
            data: List of float values representing one signal/sample

        Returns:
            NeaiState: Learning status:
                - NEAI_LEARNING_IN_PROGRESS: Continue learning with more data
                - NEAI_LEARNING_DONE: Learning complete, ready for detection
                - NEAI_ERROR: An error occurred

        Raises:
            NotImplementedError: If model type doesn't support learning

        Example:
            # Train until learning is complete
            for signal in training_data:
                status = emulator.learn(signal)
                if status == NeaiState.NEAI_LEARNING_DONE:
                    print("Training complete!")
                    break
        """
        if self._learn_func is None:
            raise NotImplementedError(f'Learn function not available for {self.model_display_name}')
        data_array = self._get_data_array(data)
        result = self._learn_func(data_array)
        return NeaiState(result)

    def learn_lines(self, data: List[List[float]]) -> List[NeaiState]:
        """
        Train the model with multiple signals at once (anomaly detection only).

        Convenience method to process multiple signals in sequence. Each signal
        is passed to learn() individually.

        Args:
            data: List of signals, where each signal is a list of float values

        Returns:
            List[NeaiState]: Learning status for each signal processed

        Example:
            signals = read_csv_data('training.csv')
            results = emulator.learn_lines(signals)

            # Check if learning completed
            if NeaiState.NEAI_LEARNING_DONE in results:
                print("Training finished!")
        """
        results = []
        for line in data:
            results.append(self.learn(line))
        return results

    def detect(self, data: List[float]) -> DetectionResult:
        """
        Run inference on a single signal.

        Processes one signal through the model and returns the result. The interpretation
        of the result depends on the model type:
        - Anomaly Detection: Returns similarity score (0-100, higher = more similar to learned patterns)
        - Classification: Returns predicted class ID (0 to N-1)
        - Outlier Detection: Returns confidence score
        - Extrapolation: Returns predicted continuous value

        Args:
            data: List of float values representing one signal/sample

        Returns:
            tuple: (status, result) where:
                - status: NeaiState indicating operation success/failure
                - result:
                    - int for classification/anomaly/outlier (class_id or similarity score)
                    - float for extrapolation (predicted value)

        Example:
            # Anomaly detection
            status, similarity = emulator.detect(test_signal)
            if status == NeaiState.NEAI_OK:
                if similarity > 90:
                    print(f"Nominal behavior (similarity: {similarity}%)")
                else:
                    print(f"Anomaly detected! (similarity: {similarity}%)")

            # Classification
            status, class_id = emulator.detect(test_signal)
            if status == NeaiState.NEAI_OK:
                print(f"Predicted class: {class_id}")

            # Extrapolation
            status, value = emulator.detect(test_signal)
            if status == NeaiState.NEAI_OK:
                print(f"Predicted value: {value}")
        """
        data_array = self._get_data_array(data)
        if self.model_type == 'anomaly_detection':
            # Anomaly detection: returns similarity (uint8)
            output_kpi = ctypes.c_uint8(0)
            neai_status = self._detect_func(data_array, ctypes.byref(output_kpi))
            ad_is_nominal = output_kpi.value >= ANOMALY_DETECTION_THRESHOLD
            return DetectionResult(NeaiState(neai_status), output_kpi.value, ad_is_nominal=ad_is_nominal)
        elif self.model_type == 'classification':
            # Classification: returns class_id (int)
            class_id = ctypes.c_int(0)
            output_buffer = (ctypes.c_float * self.class_number)(*([0] * self.class_number))
            neai_status = self._detect_func(data_array, output_buffer, ctypes.byref(class_id))
            return DetectionResult(NeaiState(neai_status), class_id.value, self.class_names[class_id.value])
        elif self.model_type == 'extrapolation':
            # Extrapolation: returns predicted float value
            output_value = ctypes.c_float(0)
            neai_status = self._detect_func(data_array, ctypes.byref(output_value))
            return DetectionResult(NeaiState(neai_status), output_value.value)
        elif self.model_type == 'outlier_detection':
            # Outlier detection: returns is_outlier (uint8)
            output_kpi = ctypes.c_uint8(0)
            neai_status = self._detect_func(data_array, ctypes.byref(output_kpi))
            return DetectionResult(NeaiState(neai_status), output_kpi.value)

        raise RuntimeError('Unsupported model type for detection')

    def detect_lines(self, data: List[List[float]]) -> List[DetectionResult]:
        """
        Run inference on multiple signals at once.

        Convenience method to process multiple signals in sequence. Each signal
        is passed to detect() individually.

        Args:
            data: List of signals, where each signal is a list of float values

        Returns:
            List of (status, result) tuples for each detection:
                - For classification: List[tuple[NeaiState, int]] - class IDs
                - For extrapolation: List[tuple[NeaiState, float]] - predicted values
                - For anomaly/outlier: List[tuple[NeaiState, int]] - similarity scores

        Example:
            test_signals = read_csv_data('test_data.csv')
            results = emulator.detect_lines(test_signals)

            # Process results
            for i, (status, value) in enumerate(results):
                if status == NeaiState.NEAI_OK:
                    print(f"Signal {i}: {value}")
                else:
                    print(f"Signal {i}: Error - {status.name}")
        """
        results = []
        for line in data:
            results.append(self.detect(line))
        return results

    def get_class_name(self, class_id: int) -> Optional[str]:
        """
        Get the name of a class by its ID (classification models only).

        Args:
            class_id: The class ID (0 to N-1)

        Returns:
            str: Class name, or None if function not available or invalid ID

        Example:
            for i in range(emulator.get_number_of_classes()):
                name = emulator.get_class_name(i)
                print(f"Class {i}: {name}")
        """
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return None


def read_csv_data(csv_path: str, drop_first_column: bool = False) -> List[List[float]]:
    """
    Read and parse CSV file containing signal data.

    Automatically detects the delimiter used in the CSV file (space, comma, tab, etc.)
    and returns the data as a list of signals, where each signal is a list of float values.
    Empty lines and whitespace are automatically handled.

    Args:
        csv_path: Path to the CSV file to read
        drop_first_column: Whether to skip the first column from each row

    Returns:
        List[List[float]]: List of signals, where each inner list represents one
                          complete signal/sample from a CSV row

    Raises:
        FileNotFoundError: If the CSV file cannot be found
        ValueError: If the CSV contains invalid numeric values

    Example:
        # Read training data
        signals = read_csv_data('training_data.csv')
        print(f"Loaded {len(signals)} signals")
        print(f"Each signal has {len(signals[0])} values")

        # Use with emulator
        with NanoEdgeAIEmulator('libneai.so') as emulator:
            for signal in signals:
                status, result = emulator.detect(signal)

    CSV Format:
        The CSV file should have one signal per line with values separated by
        spaces, commas, or tabs:

        1.23 4.56 7.89 10.11
        2.34 5.67 8.90 11.22
        3.45 6.78 9.01 12.33
    """
    data = []
    with open(csv_path, 'r') as f:
        sample = f.read(1024)
        f.seek(0)

        try:
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
        except csv.Error:
            delimiter = ' '

        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if drop_first_column and row:
                row_values = row[1:]
            else:
                row_values = row
            data_row = []
            for value in row_values:
                if value.strip():
                    data_row.append(float(value))
            if data_row:  # Only add non-empty rows
                data.append(data_row)
    return data


def cli_run(parser) -> None:
    """Run the NanoEdgeAI Studio Emulator from command line arguments."""

    print(f'NanoEdgeAI Studio Emulator - Python interface for NanoEdgeAI libraries - Copyright (c) 2025 STMicroelectronics\n')

    # Parse arguments
    args = parser.parse_args()

    # Initialize emulator with the provided library
    with NanoEdgeAIEmulator(args.lib, args.ad_use_embedded_knowledge) as emulator:
        print(f'Detected algorithm type: {emulator.model_display_name}')

        if emulator.ad_use_embedded_knowledge:
            print('Using embedded knowledge for Anomaly Detection model initialization.')

        # Learning phase (anomaly detection only)
        if args.learn:
            print(f'\nLearning from: {args.learn}')
            data = read_csv_data(args.learn)
            results = emulator.learn_lines(data)

            # Check if learning is complete
            if NeaiState.NEAI_LEARNING_DONE in results:
                print(f'Learning complete! ({len(results)} signals)')
            else:
                print(f'Learning still in progress after {len(results)} signals. More data may be needed.')

        # Detection/inference phase
        if args.detect:
            print(f'\nDetecting from: {args.detect}')
            data = read_csv_data(args.detect)
            results = emulator.detect_lines(data)
            nb_results = len(results)

            # Display results
            if args.verbose or emulator.model_type == 'extrapolation':
                print(f'Detection results ({nb_results} signals):')
                for i, detection_result in enumerate(results):
                    line_num = str(i + 1).rjust(4)
                    if detection_result.state == NeaiState.NEAI_OK:
                        if emulator.model_type == 'classification':
                            print(f'  Signal {line_num}: {detection_result.class_name} (Class ID {detection_result.value})')
                        elif emulator.model_type == 'anomaly_detection':
                            status_str = 'Nominal' if detection_result.ad_is_nominal else 'Anomaly'
                            print(f'  Signal {line_num}: Similarity {detection_result.value:>3}% - {status_str}')
                        else:
                            print(f'  Signal {line_num}: {detection_result.value}')
                    else:
                        print(f'  Signal {line_num}: Error - {detection_result.state.name}')

            # Display summary
            if emulator.model_type != 'extrapolation':
                rjust_nb = int(math.log10(nb_results)) + 1
                print(f'\nDetection summary ({nb_results} signals):')
                if emulator.model_type == 'anomaly_detection':
                    nominal_count = sum(1 for r in results if r.ad_is_nominal)
                    anomaly_count = nb_results - nominal_count
                    print(f'  Nominal: {nominal_count:>{rjust_nb}} signals\n  Anomaly: {anomaly_count:>{rjust_nb}} signals')
                elif emulator.model_type == 'classification':
                    class_counts: Dict[str, int] = {c: 0 for c in emulator.class_names}
                    for r in results:
                        if r.state == NeaiState.NEAI_OK:
                            class_name = r.class_name or 'Unknown'
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    for class_name, count in class_counts.items():
                        print(f'  {class_name}: {count:>{rjust_nb}} signals')
                elif emulator.model_type == 'outlier_detection':
                    outlier_count = sum(1 for r in results if r.value)
                    nominal_count = nb_results - outlier_count
                    print(f'  Nominal: {nominal_count:>{rjust_nb}} signals\n  Outlier: {outlier_count:>{rjust_nb}} signals')

            # Display error summary
            error_count = sum(1 for r in results if r.state != NeaiState.NEAI_OK)
            if error_count > 0:
                print(f'Detection errors: {error_count} signals')

        # If neither learn nor detect provided, print help
        if not args.learn and not args.detect:
            print('\nNo operation specified. Please provide --learn and/or --detect arguments.\n')
            parser.print_help()


def main() -> None:
    """
    Command-line interface for NanoEdgeAI Studio Emulator.

    Provides a convenient way to interact with NanoEdgeAI libraries from the
    command line without writing Python code. Supports learning (for anomaly
    detection) and inference operations on CSV data files.

    Usage Examples:
        # Anomaly detection - learn phase
        python nanoedgeai_studio_emulator.py --learn training.csv

        # Run inference/detection
        python nanoedgeai_studio_emulator.py --detect test_data.csv

        # Complete workflow (learn then detect)
        python nanoedgeai_studio_emulator.py --learn training.csv --detect test_data.csv

        # Classification model
        python nanoedgeai_studio_emulator.py --detect samples.csv

    Arguments:
        --lib: Path to the NanoEdgeAI library file (libneai.dll or libneai.so)
        --learn: CSV file containing training signals (anomaly detection only)
        --detect: CSV file containing test signals for inference

    Output:
        Results are printed to stdout showing the status and values returned
        by the library for each operation.
    """
    parser = argparse.ArgumentParser(
        epilog='For more information, visit: https://stm32ai.st.com/nanoedge-ai/'
    )
    parser.add_argument(
        '--learn',
        type=str,
        help='CSV file containing training signals (Anomaly Detection only)'
    )
    parser.add_argument(
        '--detect',
        type=str,
        help='CSV file containing test signals for inference/detection'
    )
    parser.add_argument(
        '--lib',
        type=str,
        help='[Optional] Path to NanoEdgeAI library file (libneai.dll on Windows, libneai.so on Linux). Auto-detected if not provided.',
        required=False
    )
    parser.add_argument(
        '--ad_use_embedded_knowledge',
        action='store_true',
        help='For Anomaly Detection models, whether to use embedded knowledge during initialization. Ignored for other model types.'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')

    try:
        cli_run(parser)
    except Exception as e:
        print(f'\nError: {e}')



if __name__ == '__main__':
    main()
