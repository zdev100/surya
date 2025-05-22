#include <iostream>
#include <string>
#include <vector>
#include <stdexcept> // For std::runtime_error

#include "image_utils.h"
#include "onnx_inference.h"
#include "layout_parser.h"  // Defines LayoutData
// #include "ocr_parser.h"  // OCR removed
#include "json_utils.h"
#include "visualization_utils.h"
#include <opencv2/core/mat.hpp> // For cv::Mat
#include <opencv2/imgcodecs.hpp> // For cv::imread, indirectly used by ImageUtils::loadImage
#include <fstream> // For std::ifstream for file existence check

// LayoutData is defined in layout_parser.h

// Helper function to check if a file exists
bool fileExists(const std::string& file_path) {
    std::ifstream f(file_path.c_str());
    return f.good();
}

// Helper function to derive output file paths
std::string getOutputJsonPath(const std::string& image_path) {
    size_t dot_pos = image_path.find_last_of(".");
    std::string base_name = (dot_pos == std::string::npos) ? image_path : image_path.substr(0, dot_pos);
    return base_name + "_output.json";
}

std::string getOutputImagePath(const std::string& image_path) {
    size_t dot_pos = image_path.find_last_of(".");
    std::string base_name = (dot_pos == std::string::npos) ? image_path : image_path.substr(0, dot_pos);
    return base_name + "_visualization.png"; // Or .jpg
}


int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) { // image_path, layout_model_path, optional -v
        std::cerr << "Usage: " << argv[0] << " <path_to_image> <path_to_layout_model.onnx> [-v]" << std::endl;
        std::cerr << "  <path_to_image>: Path to the input image file." << std::endl;
        std::cerr << "  <path_to_layout_model.onnx>: Path to the ONNX layout detection model." << std::endl;
        std::cerr << "  [-v]: Optional. Enable verbose logging." << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    std::string layout_model_path = argv[2];
    bool verbose = false;
    if (argc == 4 && std::string(argv[3]) == "-v") {
        verbose = true;
        std::cout << "Verbose mode enabled." << std::endl;
    }

    // --- File Existence Checks ---
    if (!fileExists(image_path)) {
        std::cerr << "Error: Input image file not found: " << image_path << std::endl;
        return 1;
    }
    if (!fileExists(layout_model_path)) {
        std::cerr << "Error: Layout ONNX model file not found: " << layout_model_path << std::endl;
        return 1;
    }
    // Recognition model check removed

    // Determine output paths
    std::string output_json_path = getOutputJsonPath(image_path);
    std::string output_viz_image_path = getOutputImagePath(image_path);

    try {
        // 1. Initialize ONNX Runtime Environment and Session for Layout Model
        bool use_cuda = false; // Set to true if CUDA is available and desired
        if (verbose) std::cout << "Initializing ONNX Runtime environment and loading layout model..." << std::endl;
        ONNXInferenceManager inference_manager(layout_model_path, use_cuda, verbose);
        if (verbose) std::cout << "ONNX Runtime environment and layout session initialized successfully." << std::endl;

        // 2. Load Image
        if (verbose) std::cout << "Loading image: " << image_path << "..." << std::endl;
        cv::Mat original_image = ImageUtils::loadImage(image_path);
        if (original_image.empty()) {
            std::cerr << "Error loading image: " << image_path << std::endl;
            return 1;
        }
        std::cout << "Image loaded successfully: " << image_path << " (" << original_image.cols << "x" << original_image.rows << ")" << std::endl;

        if (verbose) std::cout << "Image loaded successfully: " << image_path << " (" << original_image.cols << "x" << original_image.rows << ")" << std::endl;

        // 3. Process Layout Analysis
        // These node names are examples and must match your ONNX model's input/output names
        // TODO: Make these configurable or discoverable from the model
        const char* layout_input_node_name = "input_image"; 
        std::vector<const char*> layout_output_node_names = {"output_segmentation_map"}; 

        if (verbose) std::cout << "Starting layout processing..." << std::endl;
        LayoutData layout_results = processLayout(
            original_image,
            inference_manager.getLayoutSession(),
            inference_manager.getAllocator(),
            layout_input_node_name,
            layout_output_node_names,
            verbose
        );
        if (verbose) std::cout << "Layout processing completed. Detected " << layout_results.detected_polygons.size() << " layout regions." << std::endl;
        else std::cout << "Detected " << layout_results.detected_polygons.size() << " layout regions." << std::endl;


        // OCR processing removed

        // 5. Generate JSON Output for Layout
        if (verbose) std::cout << "Generating JSON output for layout..." << std::endl;
        nlohmann::json layout_json_output = JsonUtils::layoutDataToJson( // This function should now be the primary JSON generator
            layout_results, image_path, original_image.cols, original_image.rows // Pass metadata here
        );

        if (JsonUtils::saveJsonToFile(layout_json_output, output_json_path)) {
            std::cout << "Layout JSON output successfully saved to: " << output_json_path << std::endl;
        } else {
            std::cerr << "Error: Failed to save layout JSON output to: " << output_json_path << std::endl;
        }

        // 6. Generate Visualization for Layout
        if (verbose) std::cout << "Generating layout visualization image..." << std::endl;
        if (VisualizationUtils::drawLayout(original_image, layout_results, output_viz_image_path)) { // Changed function name
            std::cout << "Layout visualization image successfully saved to: " << output_viz_image_path << std::endl;
        } else {
            std::cerr << "Error: Failed to save layout visualization image to: " << output_viz_image_path << std::endl;
        }

        std::cout << "Layout application finished successfully." << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Critical Error: " << e.what() << std::endl;
        std::cerr << "OrtErrorCode: " << e.GetOrtErrorCode() << std::endl;
        std::cerr << "ErrorCategory: " << e.GetErrorCategory() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occurred." << std::endl;
        return 1;
    }

    return 0;
}
