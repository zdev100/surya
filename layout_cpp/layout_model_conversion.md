# Focused Guide: Converting EfficientViTForSemanticSegmentation (Layout Model) to ONNX

This guide provides specific instructions and considerations for converting the `EfficientViTForSemanticSegmentation` PyTorch model, used for layout detection in this project, to the ONNX (Open Neural Network Exchange) format.

For general principles of PyTorch to ONNX conversion, common challenges, and more detailed explanations of `torch.onnx.export()` parameters, please refer to the main `pytorch_to_onnx_guide.md` document.

## 1. Model Loading

The first step is to load your pre-trained `EfficientViTForSemanticSegmentation` model. You can typically do this using the project's `DetectionModelLoader` (if available and configured for your checkpoint) or directly via the Hugging Face Transformers library if you have the model identifier or path.

It is **critical** to set the model to evaluation mode (`model.eval()`) before export. This disables operations like dropout and ensures batch normalization layers use running statistics, which is essential for consistent inference.

**Python Pseudo-code:**

```python
import torch
from surya.detection.loader import DetectionModelLoader # Assuming this is the project's loader
# Alternatively, if loading directly from Hugging Face Transformers:
# from transformers import EfficientViTForSemanticSegmentation, EfficientViTImageProcessor, EfficientViTConfig

# --- Option 1: Using Project's DetectionModelLoader ---
# Ensure your project settings (e.g., settings.DETECTOR_MODEL_CHECKPOINT) point to your trained model
# loader = DetectionModelLoader() # Or DetectionModelLoader(checkpoint="path/to/your/checkpoint.pth")
# model = loader.model
# processor = loader.processor
# model.eval() # The loader might do this, but an explicit call is good practice.

# --- Option 2: Loading directly via Hugging Face Transformers ---
# model_name_or_path = "your-hf-model-identifier" or "/path/to/your/local/hf_model_directory"
# config = EfficientViTConfig.from_pretrained(model_name_or_path)
# model = EfficientViTForSemanticSegmentation.from_pretrained(model_name_or_path, config=config)
# processor = EfficientViTImageProcessor.from_pretrained(model_name_or_path) # Or your specific processor
# model.eval() # Crucial step!

# Ensure the model is on the desired device (e.g., CPU for export, or GPU if testing there)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

print("Model loaded and set to evaluation mode.")
```

## 2. Example Input Tensor

ONNX export works by tracing the model with an example input. This input must match the shape, data type, and device that the model expects.

*   **Shape**: The input shape for `EfficientViTForSemanticSegmentation` is typically `(batch_size, num_channels, height, width)`.
    *   `batch_size`: Usually 1 for export, then made dynamic.
    *   `num_channels`: 3 for RGB images.
    *   `height`, `width`: These dimensions are often determined by the model's configuration or the associated image processor (e.g., `EfficientViTImageProcessor` or `SegformerImageProcessor` from the `surya.detection.processor` module). Common sizes might be 512x512, but check your specific model's training configuration.
*   **Data Type**: Typically `torch.float32`.
*   **Device**: Should match the model's device.

**Python Pseudo-code:**

```python
import torch

# Determine these values from your model's config or processor:
# Example: if using 'processor' from the model loading step above
# target_size = processor.size # This is often a dict like {'height': 512, 'width': 512}
# height = target_size.get('height', 512) # Default to 512 if not found
# width = target_size.get('width', 512)   # Default to 512 if not found

# Or, if you know the specific dimensions:
batch_size = 1
num_channels = 3
height = 512  # Replace with your model's actual expected height
width = 512   # Replace with your model's actual expected width

# Create a dummy input tensor
# Ensure it's on the same device as the model
# example_input = torch.randn(batch_size, num_channels, height, width, device=model.device)
example_input = torch.randn(batch_size, num_channels, height, width) # Assuming model is on CPU for this example

print(f"Example input tensor created with shape: {example_input.shape} and type: {example_input.dtype}")
```

## 3. `torch.onnx.export()` Parameters

With the model and example input ready, you can call `torch.onnx.export()`. Pay close attention to the following parameters for the layout model:

*   **`input_names`**: A list of names for the input nodes in the ONNX graph. `['input_image']` is a sensible choice.
*   **`output_names`**: A list of names for the output nodes. For `EfficientViTForSemanticSegmentation`, the primary output is usually logits for each class at each pixel. A name like `['output_segmentation_logits']` or simply `['logits']` is common. Inspect your model's forward pass if unsure.
*   **`dynamic_axes`**: This is highly recommended for flexibility.
    *   For `input_image`: Make batch size, height, and width dynamic.
    *   For `output_segmentation_logits`: Batch size is typically dynamic. The height and width of the output logits often depend on the input height/width and model architecture (e.g., they might be `input_height / stride` and `input_width / stride`). You can make these dynamic too.
*   **`opset_version`**: Start with a modern opset (e.g., 12, 14, or higher). If export fails due to unsupported operators, you might need to adjust this or simplify the model. Ensure the chosen opset is supported by the ONNX Runtime version you plan to use in C++.
*   **`export_params`**: Set to `True` to include trained parameters (weights) in the ONNX file.
*   **`do_constant_folding`**: Set to `True` for optimizations.

**Python Pseudo-code Example:**

```python
import torch

# --- Assuming 'model' and 'example_input' are defined as above ---

output_onnx_path = "efficientvit_layout_model.onnx"
input_names = ["input_image"]
output_names = ["output_segmentation_logits"] # Or just "logits"

# Define dynamic axes
# The output height and width might be different from input H, W depending on the model's downsampling.
# If the output logits map directly to input pixels (H,W) after some processing,
# you might only need batch_size to be dynamic for the output.
# If the output spatial dimensions are also variable with input H,W, make them dynamic.
# Example: if output H/W is input H/W divided by a fixed stride (e.g. 4)
# 'output_height_divided_by_4', 'output_width_divided_by_4' could be names for these dynamic dimensions.

dynamic_axes = {
    input_names[0]: {0: 'batch_size', 2: 'height', 3: 'width'},
    output_names[0]: {0: 'batch_size'} 
    # If output height/width are also dynamic and tied to input H/W:
    # output_names[0]: {0: 'batch_size', 2: 'output_H', 3: 'output_W'}
}

opset_version = 14 # Try 12, 14, or newer. Check compatibility with your target ONNX Runtime.

print(f"Exporting model to ONNX at: {output_onnx_path}")
torch.onnx.export(
    model,
    example_input,
    output_onnx_path,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=opset_version,
    export_params=True,         # Store the trained parameter weights inside the model file
    do_constant_folding=True,   # Execute constant folding for optimization
    verbose=False               # Set to True for detailed ONNX export logging
)
print("Model export complete.")
```

## 4. Verifying the Conversion (Briefly)

After exporting, it's crucial to verify the ONNX model:

1.  **Load the ONNX model** using ONNX Runtime in Python.
2.  **Create an inference session.**
3.  **Prepare the same `example_input`** (converted to a NumPy array).
4.  **Run inference** with both the original PyTorch model and the ONNX Runtime session.
5.  **Compare the outputs**. They should be numerically very close (allowing for minor floating-point differences).

```python
# import onnxruntime
# import numpy as np

# ort_session = onnxruntime.InferenceSession(output_onnx_path)

# # Prepare input for ONNX Runtime (must be NumPy array)
# ort_inputs = {input_names[0]: example_input.cpu().numpy()}

# # Run ONNX inference
# ort_outputs = ort_session.run(output_names, ort_inputs)
# onnx_output_tensor = ort_outputs[0]

# # Run PyTorch inference (if not already done)
# with torch.no_grad():
#     pytorch_output_tensor = model(example_input)
#     # The model might return a dict or a custom output object.
#     # Extract the relevant tensor, e.g., pytorch_output_tensor = pytorch_output_tensor.logits
#     pytorch_output_tensor = pytorch_output_tensor.cpu().numpy() # Ensure it's a NumPy array on CPU

# # Compare
# try:
#     np.testing.assert_allclose(pytorch_output_tensor, onnx_output_tensor, rtol=1e-03, atol=1e-05)
#     print("ONNX model output matches PyTorch model output successfully.")
# except AssertionError as e:
#     print(f"Output mismatch error: {e}")

```

## 5. Crucial Note on C++ Parsing (layout_cpp/layout_parser.cpp)

The C++ application (`LayoutApp`) needs to correctly interpret the output tensor(s) from your converted ONNX layout model. The name suggested above, `output_segmentation_logits` (or `logits`), typically implies that the model outputs raw scores for different semantic classes (e.g., background, text region, image region) for each pixel or patch in the output feature map.

**The current ONNX output parsing logic in `layout_cpp/layout_parser.cpp` is a SIMPLIFIED PLACEHOLDER.** It generates dummy geometric data and does **NOT** perform the necessary steps to convert raw logits into actual layout polygons.

To get meaningful layout polygons from the `output_segmentation_logits` tensor in C++, you will need to implement steps such as:

1.  **Accessing Tensor Data**: Get a pointer to the float data from the `Ort::Value` output tensor.
2.  **Understanding Shape**: The shape will likely be `(batch_size, num_classes, output_height, output_width)`.
3.  **Post-processing**:
    *   **Argmax/Softmax (Optional but common)**: If your model has multiple classes per pixel, you might apply a softmax across the class dimension and then an argmax to get the most likely class ID for each pixel. Or, if it's a binary segmentation (text vs. background), you might just take the logits for the "text" class.
    *   **Sigmoid (If binary or multi-label)**: If the output is logits for a single class (e.g., "text") or multiple independent classes, apply a sigmoid function to convert them to probabilities (0 to 1).
    *   **Thresholding**: Apply a threshold (e.g., 0.5) to the probabilities to create a binary mask (1 for text, 0 for background).
    *   **Contour Finding**: Use OpenCV's `cv::findContours` on the binary mask to find contiguous regions.
    *   **Polygon Approximation**: Convert contours to polygons (e.g., using `cv::approxPolyDP` or `cv::minAreaRect` if rectangular boxes are sufficient).
    *   **Scaling**: Scale the polygon coordinates from the model's output dimensions (e.g., 512x512) back to the original input image's dimensions.

**You MUST replace the placeholder parsing logic in `layout_cpp/layout_parser.cpp` with these (or similar) steps tailored to your specific model's output format to extract meaningful layout information.**

Refer to the Python postprocessing logic in `surya.detection.heatmap.HeatmapProcessor.polygons_from_heatmap` or similar functions in the original Surya project for an example of how these logits are typically processed to derive polygons. This logic will need to be translated to C++.
```
