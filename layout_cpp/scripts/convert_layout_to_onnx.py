import torch
import argparse
import os
import sys

try:
    # Attempt to import from the surya package
    from surya.detection.loader import DetectionModelLoader
    from surya.detection.model import EfficientViTForSemanticSegmentation # For type hinting and potential direct use
    from surya.settings import DETECTOR_MODEL_CHECKPOINT, DETECTOR_CONFIG_CHECKPOINT
    from surya.detection.processor import SegformerImageProcessor # To understand input processing if needed
except ImportError:
    print("Error: The 'surya' library is not installed or not found in the Python path.")
    print("Please ensure 'surya' is installed (e.g., pip install surya-ocr) and accessible.")
    # Add the project root to sys.path if running from a specific directory within the project
    # This is a common workaround if the package isn't installed in editable mode.
    # For example, if 'scripts' is one level down from the project root:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from surya.detection.loader import DetectionModelLoader
        from surya.detection.model import EfficientViTForSemanticSegmentation
        from surya.settings import DETECTOR_MODEL_CHECKPOINT, DETECTOR_CONFIG_CHECKPOINT
        from surya.detection.processor import SegformerImageProcessor
        print(f"Successfully imported 'surya' components after adding {project_root} to sys.path.")
    except ImportError:
        print("Error: Failed to import 'surya' components even after attempting to modify sys.path.")
        print("Ensure your project structure is correct and 'surya' is discoverable.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch Layout Model (EfficientViTForSemanticSegmentation) to ONNX.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the PyTorch model checkpoint (.pth) or Hugging Face model name/path (e.g., 'vikp/surya_layout'). "
             "If not provided, uses the default from surya.settings.DETECTOR_MODEL_CHECKPOINT."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the model config file, required if checkpoint_path is a local .pth file and not an HF model. "
             "If not provided, uses default from surya.settings.DETECTOR_CONFIG_CHECKPOINT."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="layout_model.onnx", # Default to current dir, will be prefixed by script dir later
        help="Path to save the output ONNX model. Default: layout_model.onnx in the script's directory."
    )
    parser.add_argument(
        "--img_size",
        type=str,
        default="512,512",
        help='Comma-separated height,width for the example input image (e.g., "512,512"). Default: "512,512".'
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the example input tensor. Default: 1."
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=14, # Common modern choice
        help="ONNX opset version. Default: 14."
    )

    args = parser.parse_args()

    # Ensure output directory exists (if specified as part of output_path)
    # and adjust default output_path to be in the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output_path == "layout_model.onnx": # Default value used
        args.output_path = os.path.join(script_dir, args.output_path)
    
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    try:
        # 1. Model Loading
        print("Loading layout model...")
        checkpoint_to_load = args.checkpoint_path if args.checkpoint_path else DETECTOR_MODEL_CHECKPOINT
        config_to_load = args.config_path if args.config_path else DETECTOR_CONFIG_CHECKPOINT

        if not checkpoint_to_load:
            print("Error: No checkpoint_path provided and DETECTOR_MODEL_CHECKPOINT in surya.settings is not set.")
            sys.exit(1)
        
        # DetectionModelLoader expects checkpoint and config to be passed if not using defaults
        # and its constructor can take these directly.
        loader_params = {}
        if args.checkpoint_path: # User override for checkpoint
            loader_params['checkpoint'] = args.checkpoint_path
        if args.config_path: # User override for config
             loader_params['config'] = args.config_path
        
        print(f"Using checkpoint: {loader_params.get('checkpoint', DETECTOR_MODEL_CHECKPOINT)}")
        print(f"Using config: {loader_params.get('config', DETECTOR_CONFIG_CHECKPOINT)}")

        model_loader = DetectionModelLoader(**loader_params)
        model = model_loader.model
        # processor = model_loader.processor # For reference, not directly used for dummy tensor shape here

        # Ensure model is in evaluation mode (loader should handle this, but good to be explicit)
        model.eval()
        # Determine device (loader should also handle this)
        device = next(model.parameters()).device
        print(f"Model loaded successfully on device: {device}. Model is in evaluation mode.")

        # 2. Example Input Tensor Preparation
        print("Preparing example input tensor...")
        try:
            img_height, img_width = map(int, args.img_size.split(','))
            if len(args.img_size.split(',')) != 2:
                raise ValueError
        except ValueError:
            print(f"Error: Invalid img_size format '{args.img_size}'. Expected 'height,width' (e.g., '512,512').")
            sys.exit(1)

        example_input = torch.randn(args.batch_size, 3, img_height, img_width).to(device)
        print(f"Example input tensor created with shape: {example_input.shape}, type: {example_input.dtype}, on device: {example_input.device}")

        # 3. ONNX Export Configuration
        input_names = ['input_image']
        # The output name should match what the model's forward pass returns or what you expect.
        # For EfficientViTForSemanticSegmentation, the output is typically an object with a 'logits' attribute.
        # During tracing, ONNX will capture the actual tensor output.
        output_names = ['output_segmentation_logits'] # Or simply 'logits'

        dynamic_axes = {
            input_names[0]: {0: 'batch_size', 2: 'height', 3: 'width'},
            output_names[0]: {0: 'batch_size'} 
            # If the output height/width are also dynamic and tied to input H/W (e.g. H/stride, W/stride):
            # Add: output_names[0]: {0: 'batch_size', 2: 'output_height', 3: 'output_width'}
            # For now, we assume the C++ side will handle potentially different output H/W.
        }
        opset_version = args.opset_version
        print(f"ONNX export configuration: input_names={input_names}, output_names={output_names}, opset_version={opset_version}")
        print(f"Dynamic axes: {dynamic_axes}")

        # 4. torch.onnx.export() Call
        print(f"Exporting model to ONNX: {args.output_path}...")
        torch.onnx.export(
            model,
            example_input,
            args.output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            export_params=True,
            do_constant_folding=True,
            verbose=False # Set to True for very detailed ONNX export steps
        )
        print(f"ONNX model successfully exported to: {args.output_path}")

    except FileNotFoundError as e:
        print(f"Error: File not found. This could be the checkpoint, config, or an issue with surya.settings paths.")
        print(f"Details: {e}")
        sys.exit(1)
    except ImportError as e:
        # This might catch issues if a specific model class isn't found by the loader
        print(f"Error: Import issue during model loading or ONNX export setup.")
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
