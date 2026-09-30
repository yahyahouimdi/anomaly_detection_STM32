# NanoEdgeAI Studio Emulator

Python wrapper for **any** `libneai` library compiled by NanoEdgeAI Studio.  
This script can load and execute **any** `libneai.dll` (Windows) or `libneai.so` (Linux) regardless of the model type — no recompilation or configuration needed.

> Copyright (c) 2025 STMicroelectronics — All rights reserved.  
> For more information: https://stm32ai.st.com/nanoedge-ai/

---

## Requirements

- Python 3.7+
- `ctypes` (standard library, no install needed)
- A compiled NanoEdgeAI library: `libneai.dll` (Windows) or `libneai.so` (Linux)

---

## Supported Model Types

The emulator automatically detects the model type from the library:

| Model Type          | Learn | Detect |
|---------------------|-------|--------|
| Anomaly Detection   | ✅    | ✅     |
| Classification      | ❌    | ✅     |
| Outlier Detection   | ❌    | ✅     |
| Extrapolation       | ❌    | ✅     |

---

## Command Line Usage

```bash
python nanoedgeai_studio_emulator.py [--lib path/to/libneai] [--learn train.csv] [--detect test.csv] [-v]
```

### Arguments

| Argument                    | Description                                                                                     |
|-----------------------------|-------------------------------------------------------------------------------------------------|
| `--lib`                     | *(Optional)* Path to the `libneai` library file. Auto-detected in the script directory if omitted. |
| `--learn <file.csv>`        | CSV file with training signals. **Anomaly Detection only.**                                     |
| `--detect <file.csv>`       | CSV file with signals to run inference on.                                                      |
| `--ad_use_embedded_knowledge` | Use embedded knowledge during Anomaly Detection initialization. Ignored for other model types. |
| `-v`, `--verbose`           | Enable verbose output (prints result for each signal).                                          |

### Examples

```bash
# Anomaly detection — learn phase only
python nanoedgeai_studio_emulator.py --lib libneai.so --learn training.csv

# Detection/inference only
python nanoedgeai_studio_emulator.py --lib libneai.so --detect test.csv

# Full workflow: learn then detect
python nanoedgeai_studio_emulator.py --lib libneai.so --learn training.csv --detect test.csv

# Classification model
python nanoedgeai_studio_emulator.py --lib libneai.so --detect samples.csv --verbose

# Anomaly detection with embedded knowledge
python nanoedgeai_studio_emulator.py --lib libneai.so --learn training.csv --detect test.csv --ad_use_embedded_knowledge
```

---

## CSV Format

One signal per line, values separated by spaces, commas, or tabs (auto-detected).

```
1.23 4.56 7.89 10.11
2.34 5.67 8.90 11.22
3.45 6.78 9.01 12.33
```

---

## Python API Usage

```python
from nanoedgeai_studio_emulator import NanoEdgeAIEmulator, read_csv_data, NeaiState

with NanoEdgeAIEmulator('path/to/libneai.so') as emulator:
    print(f'Model type: {emulator.model_type}')

    # Anomaly Detection — learn phase
    if emulator.model_type == 'anomaly_detection':
        learning_data = read_csv_data('training.csv')
        for signal in learning_data:
            status = emulator.learn(signal)
            if status == NeaiState.NEAI_LEARNING_DONE:
                print('Learning complete!')
                break

    # Inference — all model types
    test_data = read_csv_data('test.csv')
    for signal in test_data:
        result = emulator.detect(signal)
        if result.state == NeaiState.NEAI_OK:
            print(f'Result: {result.value}')

    # Classification — get class names
    if emulator.model_type == 'classification':
        for i in range(emulator.class_number):
            print(f'Class {i}: {emulator.get_class_name(i)}')
```

### `DetectionResult` fields

| Field          | Type              | Description                                                       |
|----------------|-------------------|-------------------------------------------------------------------|
| `state`        | `NeaiState`       | Operation status (`NEAI_OK`, `NEAI_ERROR`, …)                    |
| `value`        | `int` or `float`  | Similarity score, class ID, outlier flag, or extrapolated value   |
| `class_name`   | `str` or `None`   | Predicted class name *(classification only)*                      |
| `ad_is_nominal`| `bool` or `None`  | `True` if similarity ≥ 90% *(anomaly detection only)*            |

### `NeaiState` values

| State                    | Meaning                                      |
|--------------------------|----------------------------------------------|
| `NEAI_OK`                | Operation completed successfully             |
| `NEAI_ERROR`             | Generic error                                |
| `NEAI_NOT_INITIALIZED`   | Library not initialized                      |
| `NEAI_INVALID_PARAM`     | Invalid parameters                           |
| `NEAI_NOT_SUPPORTED`     | Operation not supported for this model type  |
| `NEAI_LEARNING_DONE`     | Learning phase complete *(anomaly detection)*|
| `NEAI_LEARNING_IN_PROGRESS` | Learning still in progress              |

