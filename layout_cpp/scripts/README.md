# ONNX Model Conversion and Validation Scripts

## 1. Overview

This directory contains Python scripts to assist with the Surya layout detection model:

*   **`convert_layout_to_onnx.py`**: Converts the PyTorch-based layout detection model (`EfficientViTForSemanticSegmentation`) into the ONNX format. This ONNX model can then be used by the C++ layout detection application (`LayoutApp`).
*   **`validate_onnx_layout_model.py`**: Validates the accuracy of the converted ONNX layout model by comparing its output against the original PyTorch model using either a dummy input tensor or a real image.

## 2. Prerequisites

Before running these scripts, ensure you have the following installed:

*   **Python**: Version 3.8 or newer is recommended.
*   **PyTorch**: The version compatible with the Surya model checkpoints.
*   **ONNX**: `pip install onnx`
*   **ONNX Runtime**: `pip install onnxruntime` (or `onnxruntime-gpu` if you intend to use GPU for validation, though the scripts currently default to CPU for ONNX Runtime).
*   **NumPy**: `pip install numpy`
*   **Pillow (PIL)**: `pip install Pillow`
*   **The `surya` Project**:
    *   These scripts import modules directly from the `surya` project (e.g., `surya.detection.loader`, `surya.settings`).
    *   Therefore, the `surya` project's root directory must be discoverable by Python. This can be achieved by:
        1.  **Installing `surya`**: If you have installed the `surya` package (e.g., via `pip install -e .` from the project root), it should be discoverable.
        2.  **Setting `PYTHONPATH`**: Add the root directory of the `surya` project to your `PYTHONPATH` environment variable.
        3.  **Running from Project Root**: The simplest method is often to run these scripts from the root directory of the `surya` project, assuming the `layout_cpp` directory is within it. For example, if your project root is `~/surya-project/`, and the scripts are in `~/surya-project/layout_cpp/scripts/`, you would run them from `~/surya-project/`. The scripts include logic to attempt to add the parent-of-parent directory to `sys.path` as a fallback.

## 3. Script: `convert_layout_to_onnx.py`

**Purpose**: Converts the PyTorch layout detection model (`EfficientViTForSemanticSegmentation`) to the ONNX format.

**Command-Line Arguments**:

*   `--checkpoint_path PATH_OR_HF_NAME`:
    *   Path to the PyTorch model checkpoint (`.pth` file) or a Hugging Face model name/path (e.g., `vikp/surya_layout`).
    *   If not provided, the script defaults to using the checkpoint specified in `surya.settings.DETECTOR_MODEL_CHECKPOINT`.
*   `--config_path PATH_TO_CONFIG`:
    *   Path to the model's JSON configuration file. This is typically required if `--checkpoint_path` points to a local `.pth` file that isn't structured as a full Hugging Face pretrained model directory.
    *   If not provided, defaults to `surya.settings.DETECTOR_CONFIG_CHECKPOINT`.
*   `--output_path PATH_TO_ONNX_MODEL`:
    *   Path where the converted ONNX model will be saved.
    *   Default: `layout_model.onnx` (saved in the same directory as the script, i.e., `layout_cpp/scripts/`).
*   `--img_size "HEIGHT,WIDTH"`:
    *   Comma-separated height and width for the example input image used during ONNX export (e.g., `"512,512"`).
    *   Default: `"512,512"`.
*   `--batch_size INT`:
    *   Batch size for the example input tensor.
    *   Default: `1`.
*   `--opset_version INT`:
    *   The ONNX opset version to use for the export.
    *   Default: `14`.

**Example Usage**:

```bash
# Ensure you are in the root directory of the surya project, or that surya is in PYTHONPATH
python layout_cpp/scripts/convert_layout_to_onnx.py \
    --checkpoint_path vikp/surya_layout \
    # Or for a local .pth file:
    # --checkpoint_path path/to/your/layout_model.pth \
    # --config_path path/to/your/layout_config.json \ 
    --output_path layout_cpp/scripts/layout_model.onnx \
    --img_size 512,512 \
    --opset_version 14
```

## 4. Script: `validate_onnx_layout_model.py`

**Purpose**: Validates the converted ONNX layout model by comparing its output tensor(s) against the original PyTorch model's output for a given input.

**Command-Line Arguments**:

*   `--pytorch_checkpoint_path PATH_OR_HF_NAME`:
    *   Path to the PyTorch model checkpoint (`.pth`) or Hugging Face model name/path.
    *   Defaults to `surya.settings.DETECTOR_MODEL_CHECKPOINT`.
*   `--pytorch_config_path PATH_TO_CONFIG`:
    *   Path to the PyTorch model's JSON configuration file.
    *   Defaults to `surya.settings.DETECTOR_CONFIG_CHECKPOINT`.
*   `--onnx_model_path PATH_TO_ONNX_MODEL`:
    *   Path to the converted ONNX model file that needs to be validated.
    *   Default: `layout_model.onnx` (expected in the script's directory).
*   `--image_path PATH_TO_IMAGE` (Optional):
    *   Path to an actual image file to use for validation.
    *   If not provided, a dummy random tensor (matching `--img_size`) will be used as input.
*   `--img_size "HEIGHT,WIDTH"`:
    *   Comma-separated height and width for the input image. Used if `--image_path` is not provided, or to configure the processor if a real image is used.
    *   Default: `"512,512"`.
*   `--atol FLOAT`:
    *   Absolute tolerance for `numpy.allclose()` when comparing outputs.
    *   Default: `1e-4`.
*   `--rtol FLOAT`:
    *   Relative tolerance for `numpy.allclose()` when comparing outputs.
    *   Default: `1e-4`.

**Example Usage**:

*   **Using a dummy random tensor for input**:
    ```bash
    # Ensure you are in the root directory of the surya project, or that surya is in PYTHONPATH
    python layout_cpp/scripts/validate_onnx_layout_model.py \
        --pytorch_checkpoint_path vikp/surya_layout \
        --onnx_model_path layout_cpp/scripts/layout_model.onnx \
        --img_size 512,512 
    ```

*   **Using a real image for input**:
    ```bash
    # Ensure you are in the root directory of the surya project, or that surya is in PYTHONPATH
    python layout_cpp/scripts/validate_onnx_layout_model.py \
        --pytorch_checkpoint_path vikp/surya_layout \
        --onnx_model_path layout_cpp/scripts/layout_model.onnx \
        --image_path path/to/your/sample_image.jpg \
        --img_size 512,512 
    ```

## 5. Notes

*   **Paths**: Always adjust the paths to models, images, and output locations according to your specific project structure and where you have stored these files. The examples assume the scripts are run from the project root.
*   **Validation Output**:
    *   The `validate_onnx_layout_model.py` script will print "Validation Successful" if the outputs of the PyTorch and ONNX models are numerically close within the specified tolerances (`atol`, `rtol`).
    *   If the outputs differ, it will print "Validation Failed" and also show the maximum absolute difference found between the output tensors.
    *   A successful validation provides confidence that the ONNX conversion was accurate.
*   **PYTHONPATH for `surya`**: As mentioned in the Prerequisites, the `surya` library needs to be accessible. If you encounter `ImportError` for `surya` modules, ensure that the `surya` project's root directory is in your `PYTHONPATH` environment variable or that you are running the scripts from the root of the `surya` project.
*   **Model-Specific Parameters**: The default image size (`512,512`) and opset version (`14`) are common starting points. You might need to adjust these based on the specific requirements of the layout model version you are converting.
```
