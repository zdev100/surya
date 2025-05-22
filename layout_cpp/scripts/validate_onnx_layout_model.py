import torch
import onnxruntime
import numpy as np
import argparse
import os
import sys
from PIL import Image

try:
    # Attempt to import from the surya package
    from surya.detection.loader import DetectionModelLoader
    from surya.settings import DETECTOR_MODEL_CHECKPOINT, DETECTOR_CONFIG_CHECKPOINT
    from surya.detection.processor import SegformerImageProcessor
    from surya.input.processing import prepare_image_input # For image preprocessing
except ImportError:
    print("Error: Some 'surya' library components are not installed or not found in the Python path.")
    print("Please ensure 'surya' is installed (e.g., pip install surya-ocr) and accessible.")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from surya.detection.loader import DetectionModelLoader
        from surya.settings import DETECTOR_MODEL_CHECKPOINT, DETECTOR_CONFIG_CHECKPOINT
        from surya.detection.processor import SegformerImageProcessor
        from surya.input.processing import prepare_image_input
        print(f"Successfully imported 'surya' components after adding {project_root} to sys.path.")
    except ImportError as e:
        print(f"Error: Failed to import 'surya' components even after attempting to modify sys.path: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Validate a converted ONNX layout model against its PyTorch original.")
    parser.add_argument(
        "--pytorch_checkpoint_path",
        type=str,
        default=None,
        help="Path to the PyTorch model checkpoint (.pth) or Hugging Face model name/path. "
             "Defaults to surya.settings.DETECTOR_MODEL_CHECKPOINT."
    )
    parser.add_argument(
        "--pytorch_config_path",
        type=str,
        default=None,
        help="Path to PyTorch model config. Defaults to surya.settings.DETECTOR_CONFIG_CHECKPOINT."
    )
    parser.add_argument(
        "--onnx_model_path",
        type=str,
        default="layout_model.onnx",
        help="Path to the converted ONNX model. Default: layout_model.onnx in the script's directory."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to an actual image for validation. If not provided, a dummy random tensor is used."
    )
    parser.add_argument(
        "--img_size",
        type=str,
        default="512,512",
        help='Comma-separated height,width for input image (e.g., "512,512"). Default: "512,512". '
             "Used if --image_path is not provided or if the image needs specific resizing via processor."
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for numpy.allclose(). Default: 1e-4."
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for numpy.allclose(). Default: 1e-4."
    )

    args = parser.parse_args()

    # Adjust default ONNX model path to be relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.onnx_model_path == "layout_model.onnx": # Default value used
        args.onnx_model_path = os.path.join(script_dir, args.onnx_model_path)

    if not os.path.exists(args.onnx_model_path):
        print(f"Error: ONNX model file not found at {args.onnx_model_path}")
        sys.exit(1)

    try:
        # 1. Load PyTorch Model
        print("Loading PyTorch layout model...")
        checkpoint_to_load = args.pytorch_checkpoint_path if args.pytorch_checkpoint_path else DETECTOR_MODEL_CHECKPOINT
        config_to_load = args.pytorch_config_path if args.pytorch_config_path else DETECTOR_CONFIG_CHECKPOINT

        if not checkpoint_to_load:
            print("Error: No PyTorch checkpoint_path provided and DETECTOR_MODEL_CHECKPOINT in surya.settings is not set.")
            sys.exit(1)
        
        loader_params = {}
        if args.pytorch_checkpoint_path:
            loader_params['checkpoint'] = args.pytorch_checkpoint_path
        if args.pytorch_config_path:
            loader_params['config'] = args.pytorch_config_path
        
        print(f"Using PyTorch checkpoint: {loader_params.get('checkpoint', DETECTOR_MODEL_CHECKPOINT)}")
        print(f"Using PyTorch config: {loader_params.get('config', DETECTOR_CONFIG_CHECKPOINT)}")

        pytorch_model_loader = DetectionModelLoader(**loader_params)
        pytorch_model = pytorch_model_loader.model
        pytorch_processor = pytorch_model_loader.processor # SegformerImageProcessor instance

        pytorch_model.eval()
        device = next(pytorch_model.parameters()).device
        print(f"PyTorch model loaded successfully on device: {device}. Model is in evaluation mode.")

        # 2. Load ONNX Model
        print(f"Loading ONNX model from: {args.onnx_model_path}...")
        onnx_session = onnxruntime.InferenceSession(args.onnx_model_path, providers=['CPUExecutionProvider']) # Or specific providers
        print("ONNX model loaded successfully.")

        # 3. Input Preparation
        print("Preparing input tensor...")
        img_height, img_width = map(int, args.img_size.split(','))

        if args.image_path:
            if not os.path.exists(args.image_path):
                print(f"Error: Input image file not found at {args.image_path}")
                sys.exit(1)
            print(f"Loading and preprocessing image: {args.image_path}")
            image = Image.open(args.image_path).convert("RGB")
            
            # Preprocessing using surya's SegformerImageProcessor logic
            # The processor typically handles resizing to model's expected size and normalization.
            # We need to ensure the processor's size matches args.img_size or is what the model expects.
            # Forcing processor size to args.img_size if that's the reference for ONNX export.
            if isinstance(pytorch_processor, SegformerImageProcessor):
                # If ONNX model was exported with a specific size, ensure processor matches for validation
                pytorch_processor.size = {'height': img_height, 'width': img_width}
                print(f"Processor size configured to: {pytorch_processor.size}")

            # The prepare_image_input function from surya.input.processing or processor itself
            # typically takes a list of PIL Images and returns a batched tensor.
            # For a single image:
            # Option A: Using the processor directly (if it has a suitable preprocess method)
            if hasattr(pytorch_processor, 'preprocess'): # Common in HF processors
                 processed_inputs = pytorch_processor.preprocess([image], return_tensors="pt")
                 input_tensor = processed_inputs['pixel_values'].to(device)
            # Option B: Using prepare_image_input (more aligned with surya's batch processing)
            # else: # Fallback, assuming prepare_image_input is more generic
            #     input_tensor, _, _ = prepare_image_input([image], pytorch_processor, device)
            # The line above might not be fully correct, as prepare_image_input is for batching.
            # Let's assume the processor's __call__ or a direct preprocess method is best.
            # For SegformerImageProcessor, it's often its __call__ method.
            else: # Assuming processor is callable like HF processors
                input_tensor = pytorch_processor([image], return_tensors="pt")['pixel_values'].to(device)

            print(f"Image preprocessed. Tensor shape: {input_tensor.shape}")

        else:
            print(f"No image_path provided. Using dummy random tensor with shape (1, 3, {img_height}, {img_width}).")
            input_tensor = torch.randn(1, 3, img_height, img_width).to(device)
        
        numpy_input_tensor = input_tensor.cpu().numpy()

        # 4. Inference
        print("Running inference with PyTorch model...")
        with torch.no_grad():
            pytorch_outputs = pytorch_model(input_tensor)
        
        # EfficientViTForSemanticSegmentation output is an object, extract logits
        pytorch_output_np = pytorch_outputs.logits.cpu().numpy()
        print(f"PyTorch model output shape: {pytorch_output_np.shape}")

        print("Running inference with ONNX model...")
        # ONNX model input name must match the name used during export (e.g., 'input_image')
        onnx_input_name = onnx_session.get_inputs()[0].name 
        onnx_output_name = onnx_session.get_outputs()[0].name # Assuming single output
        print(f"ONNX session input name: {onnx_input_name}, output name: {onnx_output_name}")

        onnx_outputs = onnx_session.run([onnx_output_name], {onnx_input_name: numpy_input_tensor})
        onnx_output_np = onnx_outputs[0]
        print(f"ONNX model output shape: {onnx_output_np.shape}")

        # 5. Comparison
        print("Comparing PyTorch and ONNX outputs...")
        if pytorch_output_np.shape != onnx_output_np.shape:
            print(f"Error: Output shapes differ! PyTorch: {pytorch_output_np.shape}, ONNX: {onnx_output_np.shape}")
            sys.exit(1)

        if np.allclose(pytorch_output_np, onnx_output_np, atol=args.atol, rtol=args.rtol):
            print("Validation Successful: PyTorch and ONNX model outputs are close within tolerance.")
            max_abs_diff = np.max(np.abs(pytorch_output_np - onnx_output_np))
            print(f"Max absolute difference: {max_abs_diff:.6e}")
        else:
            print("Validation Failed: PyTorch and ONNX model outputs differ.")
            max_abs_diff = np.max(np.abs(pytorch_output_np - onnx_output_np))
            print(f"Max absolute difference: {max_abs_diff:.6e}")
            
            # Optionally, print more details on differences
            # for i in range(pytorch_output_np.shape[0]): # Iterate through batch or first few elements
            #     if not np.allclose(pytorch_output_np[i], onnx_output_np[i], atol=args.atol, rtol=args.rtol):
            #         print(f"Difference found in element/slice {i}:")
            #         # print("PyTorch slice:\n", pytorch_output_np[i, ...some slice...]) # Print a small slice
            #         # print("ONNX slice:\n", onnx_output_np[i, ...some slice...])
            #         # print("Difference slice:\n", pytorch_output_np[i, ...some slice...] - onnx_output_np[i, ...some slice...])
            #         break # Print only the first differing slice
            sys.exit(1) # Indicate failure

    except FileNotFoundError as e:
        print(f"Error: File not found. This could be the checkpoint, config, ONNX model, or image.")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

```
