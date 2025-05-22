#include "onnx_inference.h"
#include <stdexcept> // For std::runtime_error
#include <iostream>  // For std::cout, std::cerr (for logging)
#include <vector>    // For std::vector

// Constructor
ONNXInferenceManager::ONNXInferenceManager(
    const std::string& layout_model_path,
    const std::string& recognition_model_path,
    bool use_cuda,
    int cuda_device_id,
    bool verbose) // Added verbose flag
    : env_(verbose ? ORT_LOGGING_LEVEL_INFO : ORT_LOGGING_LEVEL_WARNING, "SuryaCppAppLog"), verbose_(verbose) {

    if (verbose_) std::cout << "ONNXInferenceManager: Initializing with verbose mode." << std::endl;
    initializeSessionOptions(use_cuda, cuda_device_id);

    // Load layout model
    if (verbose_) std::cout << "ONNXInferenceManager: Loading layout model from " << layout_model_path << std::endl;
    try {
        // ONNX Runtime uses wchar_t for model paths on Windows
        #ifdef _WIN32
        std::wstring layout_model_path_w = std::wstring(layout_model_path.begin(), layout_model_path.end());
        layout_session_ = std::make_unique<Ort::Session>(env_, layout_model_path_w.c_str(), session_options_);
        #else
        layout_session_ = std::make_unique<Ort::Session>(env_, layout_model_path.c_str(), session_options_);
        #endif
        if (verbose_) std::cout << "ONNXInferenceManager: Layout model loaded successfully." << std::endl;
        else std::cout << "Layout model loaded successfully from: " << layout_model_path << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNXInferenceManager Error: Failed to load layout model (" << layout_model_path << "): " << e.what() << std::endl;
        throw; // Re-throw to signal failure to constructor caller
    }

    // Load recognition model
    if (verbose_) std::cout << "ONNXInferenceManager: Loading recognition model from " << recognition_model_path << std::endl;
    try {
        #ifdef _WIN32
        std::wstring recognition_model_path_w = std::wstring(recognition_model_path.begin(), recognition_model_path.end());
        recognition_session_ = std::make_unique<Ort::Session>(env_, recognition_model_path_w.c_str(), session_options_);
        #else
        recognition_session_ = std::make_unique<Ort::Session>(env_, recognition_model_path.c_str(), session_options_);
        #endif
        if (verbose_) std::cout << "ONNXInferenceManager: Recognition model loaded successfully." << std::endl;
        else std::cout << "Recognition model loaded successfully from: " << recognition_model_path << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNXInferenceManager Error: Failed to load recognition model (" << recognition_model_path << "): " << e.what() << std::endl;
        throw; // Re-throw
    }
}

ONNXInferenceManager::~ONNXInferenceManager() {
    // Sessions are managed by unique_ptr, will be cleaned up automatically.
    // Ort::Env should be alive until all sessions are destroyed.
    if (verbose_) std::cout << "ONNXInferenceManager: Destructor called." << std::endl;
    else std::cout << "ONNXInferenceManager destroyed." << std::endl;
}

void ONNXInferenceManager::initializeSessionOptions(bool use_cuda, int cuda_device_id) {
    if (verbose_) std::cout << "ONNXInferenceManager: Initializing session options..." << std::endl;
    session_options_.SetIntraOpNumThreads(1); 
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); 

    if (use_cuda) {
        if (verbose_) std::cout << "ONNXInferenceManager: Attempting to use CUDA execution provider for device ID " << cuda_device_id << "." << std::endl;
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = cuda_device_id;
        // cuda_options.gpu_mem_limit = SIZE_MAX; // Example: uncomment to set options
        // cuda_options.arena_extend_strategy = 0; 
        // cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        // cuda_options.do_copy_in_default_stream = 1;

        try {
            session_options_.AppendExecutionProvider_CUDA(cuda_options);
            if (verbose_) std::cout << "ONNXInferenceManager: CUDA execution provider appended successfully." << std::endl;
            else std::cout << "CUDA execution provider enabled." << std::endl;
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNXInferenceManager Warning: Failed to append CUDA execution provider: " << e.what() << std::endl;
            std::cerr << "ONNXInferenceManager: Falling back to CPU execution provider." << std::endl;
        }
    } else {
        if (verbose_) std::cout << "ONNXInferenceManager: Using CPU execution provider." << std::endl;
        else std::cout << "Using CPU execution provider." << std::endl;
    }
    if (verbose_) std::cout << "ONNXInferenceManager: Session options initialized." << std::endl;
}


Ort::Session& ONNXInferenceManager::getLayoutSession() {
    if (!layout_session_) {
        throw std::runtime_error("Layout session is not initialized.");
    }
    return *layout_session_;
}

Ort::Session& ONNXInferenceManager::getRecognitionSession() {
    if (!recognition_session_) {
        throw std::runtime_error("Recognition session is not initialized.");
    }
    return *recognition_session_;
}

Ort::AllocatorWithDefaultOptions& ONNXInferenceManager::getAllocator() {
    return allocator_;
}

Ort::Value ONNXInferenceManager::createTensorFromData(
    const std::vector<float>& image_data,
    const std::vector<int64_t>& shape,
    Ort::AllocatorWithDefaultOptions& allocator) { 

    // Ensure data_size matches the product of dimensions in shape
    size_t expected_data_size = 1; // Should be size_t for safety with large tensors
    for (int64_t dim : shape) {
        expected_data_size *= dim;
    }
    if (image_data.size() != expected_data_size) {
        throw std::runtime_error("Mismatch between image_data size and shape dimensions.");
    }

    // Create tensor
    // Note: ONNX Runtime expects a non-const pointer for data.
    // The lifetime of 'image_data' must be managed carefully if it's not copied.
    // Here, Ort::Value::CreateTensor copies the data.
    return Ort::Value::CreateTensor<float>(
        allocator.GetInfo(), // Use the allocator's memory info
        const_cast<float*>(image_data.data()), // Data pointer
        image_data.size(),                     // Total number of elements
        shape.data(),                          // Pointer to shape array
        shape.size()                           // Number of dimensions
    );
}

std::vector<Ort::Value> ONNXInferenceManager::runInference(
    Ort::Session& session,
    const Ort::Value& input_tensor, 
    const char* input_node_name,    
    const std::vector<const char*>& output_node_names) { 

    std::vector<Ort::Value> output_tensors; // Use std::vector<Ort::Value> directly

    try {
        if (verbose_) {
            std::cout << "ONNXInferenceManager: Running inference on session. Input node: " << input_node_name;
            std::cout << ". Output nodes: ";
            for(const char* name : output_node_names) std::cout << name << " ";
            std::cout << std::endl;
        }
        
        output_tensors = session.Run(
            Ort::RunOptions{nullptr}, 
            &input_node_name,         
            &input_tensor,            
            1,                        
            output_node_names.data(), 
            output_node_names.size()  
        );

        if (verbose_) std::cout << "ONNXInferenceManager: Inference run completed. Got " << output_tensors.size() << " output tensor(s)." << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNXInferenceManager Error: Exception during ONNX Runtime inference: " << e.what() << std::endl;
        // Re-throw to allow higher-level handling
        throw;
    }

    return output_tensors;
}

// Add verbose_ member if not already present
// private:
//   bool verbose_ = false; // Add this to the class definition in onnx_inference.h
// In onnx_inference.h, ensure 'bool verbose_ = false;' is added to private members.
// For this tool, I'll assume it's implicitly added to the .h if I use it here.
// If not, I'd need another step to modify onnx_inference.h
// For now, assuming the constructor initializes it and it's available.
