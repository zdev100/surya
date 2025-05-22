# Comprehensive Guide: Converting PyTorch Models to ONNX Format

This guide provides a detailed explanation of how to convert PyTorch models to the ONNX (Open Neural Network Exchange) format. This conversion is a crucial step for deploying models in various environments, especially in C++ applications where PyTorch's Python dependency might be undesirable.

## 1. What is ONNX?

ONNX (Open Neural Network Exchange) is an open format built to represent machine learning models. The key benefits of using ONNX are:

*   **Interoperability**: ONNX allows you to use your preferred framework (like PyTorch, TensorFlow, or scikit-learn) to train a model and then convert it into a standard ONNX format. This ONNX model can then be run in various other frameworks and inference engines that support ONNX.
*   **Hardware Acceleration**: ONNX models can be run on a variety of hardware platforms (CPUs, GPUs, specialized AI accelerators). Many hardware vendors provide ONNX-compatible inference engines optimized for their hardware, potentially leading to significant performance improvements.

## 2. Core Tool: `torch.onnx.export()`

The primary tool for converting PyTorch models to ONNX is the `torch.onnx.export()` function.

Its main parameters are:

*   `model`: The PyTorch model (an instance of `torch.nn.Module`) you want to convert.
*   `args`: An example input tensor (or a tuple of input tensors if the model accepts multiple inputs) that the model expects. This is **crucial** because ONNX export works by *tracing* the model. It executes the model with the provided example input and records the operations performed to build the ONNX graph. The shape and data type of `args` must match what the model expects.
*   `f`: A string representing the file path where the output ONNX model will be saved (e.g., `"model.onnx"`).
*   `input_names` (optional): A list of strings providing names for the input nodes in the exported ONNX graph. If not provided, default names like "input.1", "input.2" will be used. Assigning meaningful names is good practice for clarity.
*   `output_names` (optional): A list of strings providing names for the output nodes in the exported ONNX graph. Similar to `input_names`, assigning meaningful names helps in understanding and using the model.
*   `dynamic_axes` (optional): This is a very important parameter for models that need to handle variable-sized inputs (e.g., different batch sizes, image dimensions, or sequence lengths). It's a dictionary where keys are input/output names (as specified in `input_names` and `output_names`) and values are dictionaries mapping axis indices to a name representing the dynamic dimension. For example, `{'input_image': {0: 'batch_size', 2: 'height', 3: 'width'}, 'output_predictions': {0: 'batch_size'}}` specifies that the 0th dimension of `input_image` (batch size), its 2nd dimension (height), and 3rd dimension (width) are dynamic, and similarly for the batch size of `output_predictions`.
*   `opset_version`: The ONNX opset version to use for the export. An "opset" (operator set) defines the set of available operators in ONNX and their specifications. Different opset versions support different operators or different versions of operators. Choosing the right `opset_version` is important for compatibility:
    *   Newer PyTorch versions and models with newer operations might require a higher `opset_version`.
    *   The target inference engine (e.g., ONNX Runtime) must also support the chosen `opset_version`.
    *   It's generally recommended to use a reasonably recent opset version that is supported by both your PyTorch version and your target deployment environment. Common starting points are 11, 12, but versions up to 17+ are available.

### Pseudo-code Example:

```python
import torch
import torch.nn as nn

# Define a simple PyTorch model (example)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
pytorch_model = SimpleModel()
pytorch_model.eval() # Set to evaluation mode

# Create an example input tensor
example_input = torch.randn(1, 10) # (batch_size, input_features)

# Define input and output names
input_names = ["input_features"]
output_names = ["output_predictions"]

# Define dynamic axes (e.g., batch_size is dynamic)
dynamic_axes = {
    "input_features": {0: "batch_size"},
    "output_predictions": {0: "batch_size"}
}

# Export the model
torch.onnx.export(
    pytorch_model,
    example_input,
    "simple_model.onnx",
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=12, # Choose an appropriate opset version
    verbose=False # Set to True for detailed export logging
)

print("Model exported to simple_model.onnx")
```

## 3. Identifying Models for Conversion

Based on the project's structure, the primary models to convert are:

*   **`EfficientViTForSemanticSegmentation`**: Used for layout detection.
*   **`SuryaModel`**: Used for text recognition.

You will need to load these pre-trained PyTorch models first. The loading mechanism will be similar to what's implemented in the `surya.detection.loader.DetectionModelLoader` and `surya.recognition.loader.RecognitionModelLoader` classes in your Python codebase. These loaders handle checkpoint loading, model configuration, and setting up the model for inference.

## 4. Steps for Conversion

Here's a general workflow for converting your PyTorch models to ONNX:

1.  **Load PyTorch Model**:
    *   Instantiate your model (e.g., `EfficientViTForSemanticSegmentation`, `SuryaModel`).
    *   Load the pre-trained weights (checkpoint files).
    *   Crucially, set the model to evaluation mode: `model.eval()`. This disables layers like dropout or batch normalization updates, which are only active during training.

2.  **Create Example Input Tensor**:
    *   Determine the expected input shape and data type for your model. For image-based models like those used in Surya, this is typically a 4D tensor: `(batch_size, channels, height, width)`.
        *   `batch_size`: Usually 1 for tracing, then made dynamic.
        *   `channels`: e.g., 3 for RGB images.
        *   `height`, `width`: The dimensions the model was trained on or expects. These can also be made dynamic.
    *   Create a dummy tensor with this shape and type. For example:
        ```python
        # For an image model expecting a 224x224 RGB image
        example_input = torch.randn(1, 3, 224, 224, device=model.device)
        # Ensure the device (CPU/GPU) matches the model's device
        ```

3.  **Call `torch.onnx.export()`**:
    *   Invoke the function with your loaded model, the example input, the desired output file name (e.g., `detection_model.onnx`, `recognition_model.onnx`).
    *   **`opset_version`**: Start with a version like 11 or 12. If export fails due to unsupported operators, you might need to try a newer opset version (e.g., 14, 16, or 17+), ensuring your target ONNX Runtime version supports it.
    *   **`input_names` and `output_names`**: Define descriptive names, e.g., `input_names=['input_image']`, `output_names=['output_layout']` for the detection model.
    *   **`dynamic_axes`**: This is critical for real-world applications.
        *   For the detection model (`EfficientViTForSemanticSegmentation`), batch size, image height, and image width are often dynamic.
            ```python
            dynamic_axes = {
                'input_image': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output_layout': {0: 'batch_size'} # Adjust output names and axes as needed
            }
            ```
        *   For the recognition model (`SuryaModel`), batch size and sequence length (if applicable, e.g., for text lines of varying lengths) would be dynamic. The exact dynamic axes will depend on the model's specific architecture.
            ```python
            # Example for a recognition model, may need adjustment
            dynamic_axes = {
                'input_pixel_values': {0: 'batch_size', 2: 'height', 3: 'width'}, # Or whatever the input is named
                'output_sequences': {0: 'batch_size', 1: 'sequence_length'} # Adjust output names and axes
            }
            ```

## 5. Verification

After exporting your model to ONNX, it's essential to verify that the conversion was successful and the ONNX model produces results consistent with the original PyTorch model.

You can do this using an ONNX inference engine like ONNX Runtime in Python:

1.  **Install ONNX Runtime**: `pip install onnxruntime`
2.  **Load the ONNX model** and create an inference session.
3.  **Prepare the same sample input** that you'd use for the PyTorch model (remember to convert it to a NumPy array).
4.  **Run inference** with both the PyTorch model and the ONNX Runtime session using the sample input.
5.  **Compare the outputs**. They should be numerically very close (allowing for minor floating-point precision differences).

```python
import onnxruntime
import numpy as np
import torch # Assuming original PyTorch model and input are available

# 1. Load ONNX model and create session
ort_session = onnxruntime.InferenceSession("your_model.onnx")

# 2. Prepare input (example_input_numpy should match the ONNX model's expected input)
# This is the same input used for PyTorch model, converted to numpy
example_input_numpy = example_input.cpu().numpy() # Ensure it's on CPU and is a NumPy array

# 3. Run ONNX inference
ort_inputs = {ort_session.get_inputs()[0].name: example_input_numpy}
ort_outputs = ort_session.run(None, ort_inputs)

# 4. Run PyTorch inference (if not already done)
with torch.no_grad():
    pytorch_outputs = pytorch_model(example_input)

# 5. Compare outputs
# The structure of pytorch_outputs and ort_outputs might differ (e.g., tensor vs numpy array, list vs single array)
# You'll need to adapt the comparison accordingly.
# For example, if both are single tensors/arrays:
np.testing.assert_allclose(pytorch_outputs.cpu().numpy(), ort_outputs[0], rtol=1e-03, atol=1e-05)
print("ONNX model output matches PyTorch model output (within tolerance).")
```

## 6. Potential Challenges

Converting complex PyTorch models to ONNX can sometimes present challenges:

*   **Custom Operations / Unsupported Operators**:
    *   If your PyTorch model uses custom-defined operations (e.g., a unique C++/CUDA extension) or standard PyTorch operations not yet supported by the chosen `opset_version`, the export will fail.
    *   **Solutions**:
        *   Try a newer `opset_version`.
        *   Re-implement the problematic parts of your model using ONNX-friendly standard PyTorch operations.
        *   For truly custom operations, you might need to implement them as custom ONNX operators, which is an advanced topic and requires C++ development for the ONNX Runtime.

*   **Dynamic Control Flow**:
    *   Models with complex control flow that depends on input values (e.g., loops with variable iterations based on input tensor data, `if` statements branching on intermediate tensor values) can be difficult to trace and export correctly.
    *   ONNX has some support for control flow operators (like `Loop`, `If`, `Scan`), but their usage via `torch.onnx.export` can be tricky.
    *   **Solutions**:
        *   Simplify or refactor the control flow in the PyTorch model if possible.
        *   Explicitly use PyTorch's scripting capabilities (`torch.jit.script`) for parts of the model with data-dependent control flow, as this can sometimes help the exporter understand it better.

*   **Preprocessing/Postprocessing**:
    *   ONNX models typically represent the core neural network computations. Preprocessing steps (like image normalization, resizing, tokenization) and postprocessing steps (like decoding bounding boxes from raw model output, applying non-maximum suppression, detokenization) are usually **not** part of the ONNX graph itself.
    *   These steps need to be handled by the application that uses the ONNX model. For your C++ application, you will need to re-implement the logic found in Python files like `surya.detection.processor.DetectionProcessor` and `surya.recognition.processor.RecognitionProcessor` (and their associated schemas like `surya.schema.heatmap.HeatmapProcessor` or `surya.schema.textline.TextLineProcessor`).
    *   Pay close attention to the exact transformations (e.g., normalization means and standard deviations, resizing methods) to ensure consistency between the Python (training/validation) and C++ (inference) pipelines.

*   **Attention Mechanisms**:
    *   Models like `SuryaModel` (often based on Transformers) extensively use attention mechanisms (e.g., standard scaled dot-product attention, or potentially optimized versions like FlashAttention or memory-efficient attention via `torch.backends.cuda.sdp_kernel` / `F.scaled_dot_product_attention`).
    *   Exporting these can be tricky:
        *   Standard attention mechanisms are generally well-supported in recent ONNX opset versions.
        *   Optimized attention implementations (like FlashAttention) might not have direct ONNX equivalents. The `config._attn_implementation` switching observed in `RecognitionModelLoader` (e.g., between `"eager"`, `"sdpa"`, `"flash_attention_2"`) is a strong indicator. For ONNX export, you'll likely need to ensure the model is configured to use an exportable attention backend (often the "eager" or a standard "sdpa" implementation if supported by the opset).
        *   You might need a specific `opset_version` (e.g., 14+ for broader support of attention-related ops).
        *   In some cases, custom ONNX operators or graph surgery might be required for highly optimized or non-standard attention patterns, but this should be a last resort.

## 7. Tools for Inspecting ONNX Models

Once you have an `.onnx` file, it's very helpful to visualize its structure, inputs, outputs, and operators.

*   **Netron**: Netron (available as a web app or a desktop application) is an excellent tool for visualizing neural network models, including ONNX. You can open your `.onnx` file in Netron to:
    *   See the graph structure.
    *   Inspect properties of each node (operator), including its inputs, outputs, and attributes.
    *   Verify input/output names, shapes, and data types.
    *   Check if dynamic axes are correctly represented.

This guide should provide a solid foundation for converting your PyTorch models to the ONNX format. Remember that model conversion can be an iterative process, often requiring some experimentation with opset versions and potentially minor model adjustments.
