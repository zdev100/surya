#ifndef ONNX_INFERENCE_H
#define ONNX_INFERENCE_H

#include <string>
#include <vector>
#include <memory> // For std::unique_ptr

// ONNX Runtime C++ API
// Ensure this path is correct or that onnxruntime_cxx_api.h is in the include path
#include <onnxruntime_cxx_api.h>

// Forward declaration of cv::Mat to avoid including OpenCV headers directly here if not needed
// namespace cv { class Mat; } // If we were to pass cv::Mat directly to some function here

class ONNXInferenceManager {
public:
    /**
     * @brief Constructor that initializes the ONNX Runtime environment and loads models.
     * @param layout_model_path Path to the .onnx file for the layout detection model.
     * @param recognition_model_path Path to the .onnx file for the text recognition model.
     * @param use_cuda If true, attempts to use CUDA execution provider. Falls back to CPU if CUDA is not available or fails.
     * @param cuda_device_id The ID of the CUDA device to use.
     */
    ONNXInferenceManager(const std::string& layout_model_path,
                         bool use_cuda = false, 
                         int cuda_device_id = 0,
                         bool verbose = false);

    ~ONNXInferenceManager();

    /**
     * @brief Gets a reference to the layout detection session.
     * @return Ort::Session& reference.
     */
    Ort::Session& getLayoutSession();

    // Removed getRecognitionSession()

    /**
     * @brief Gets a reference to the allocator.
     * @return Ort::AllocatorWithDefaultOptions& reference.
     */
    Ort::AllocatorWithDefaultOptions& getAllocator();

    /**
     * @brief Runs inference on a given session with a single input tensor.
     *
     * @param session The ONNX Runtime session to use.
     * @param input_tensor The input Ort::Value tensor.
     * @param input_node_name The name of the input node for the model.
     * @param output_node_names A vector of C-strings for the names of the output nodes.
     * @return A vector of Ort::Value objects representing the output tensors.
     */
    std::vector<Ort::Value> runInference(Ort::Session& session,
                                         const Ort::Value& input_tensor,
                                         const char* input_node_name, // Assuming single input name for simplicity
                                         const std::vector<const char*>& output_node_names);

    /**
     * @brief Creates an Ort::Value tensor from preprocessed image data.
     *
     * @param image_data A flat vector of floats representing the image (e.g., in CHW format).
     * @param shape A vector of int64_t representing the tensor shape (e.g., {1, C, H, W}).
     * @param allocator Ort::Allocator instance to use for tensor creation.
     * @return Ort::Value containing the tensor.
     */
    static Ort::Value createTensorFromData(const std::vector<float>& image_data,
                                           const std::vector<int64_t>& shape,
                                           Ort::AllocatorWithDefaultOptions& allocator);
private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;

    std::unique_ptr<Ort::Session> layout_session_;
    // Removed recognition_session_

    // Memory info for creating tensors (initialized once)
    Ort::AllocatorWithDefaultOptions allocator_; 

    // Helper to initialize session options, potentially with CUDA
    void initializeSessionOptions(bool use_cuda, int cuda_device_id);

    bool verbose_ = false; // Verbosity flag
};

#endif // ONNX_INFERENCE_H
