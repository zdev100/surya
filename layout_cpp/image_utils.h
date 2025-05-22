#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp> // OpenCV header

// Forward declaration for Ort::Value to avoid including onnxruntime_cxx_api.h here
// if it's not strictly necessary for function signatures alone.
// However, for returning Ort::Value, the full type is needed.
// For simplicity in this stage, we might include it, or use void* / custom struct
// if we want to hide ONNX details from this specific header.
// For now, let's assume direct use for clarity.
#include <onnxruntime_cxx_api.h> // For Ort::Value

namespace ImageUtils {

/**
 * @brief Loads an image from the specified path.
 * @param image_path Path to the image file.
 * @return cv::Mat object containing the loaded image. Returns an empty Mat if loading fails.
 */
cv::Mat loadImage(const std::string& image_path);

/**
 * @brief Preprocesses an image for the layout detection model.
 *
 * This function should implement the necessary steps like resizing, normalization,
 * and data type conversion (e.g., to float) and CHW format.
 * The exact parameters (target size, mean, std) should match those used
 * during the training of the layout model.
 *
 * @param image The input image (cv::Mat).
 * @param target_height The target height for the model.
 * @param target_width The target width for the model.
 * @param mean The mean values for normalization (per channel).
 * @param stddev The standard deviation values for normalization (per channel).
 * @return A vector of floats representing the preprocessed image data in CHW format.
 */
std::vector<float> preprocessLayoutImage(
    const cv::Mat& image,
    int target_height,
    int target_width,
    const std::vector<float>& mean = {0.485f, 0.456f, 0.406f}, // Example: ImageNet mean
    const std::vector<float>& stddev = {0.229f, 0.224f, 0.225f} // Example: ImageNet stddev
);

/**
 * @brief Preprocesses an image patch for the OCR recognition model.
 *
 * Similar to preprocessLayoutImage, but tailored for the OCR model's requirements.
 * This might involve different target sizes, normalization, or grayscale conversion.
 *
 * @param image_patch The input image patch (cv::Mat).
 * @param target_height The target height for the model.
 * @param target_width The target width for the model (can be variable, e.g., if model handles dynamic width).
 * @param mean The mean values for normalization (single value if grayscale, or per channel).
 * @param stddev The standard deviation values for normalization.
 * @param to_grayscale Whether to convert the patch to grayscale first.
 * @return A vector of floats representing the preprocessed image data.
 */
std::vector<float> preprocessOCRImage(
    const cv::Mat& image_patch,
    int target_height,
    // int target_width, // Often, OCR models handle variable width, or padding is done here.
    // For now, let's assume a fixed height and the width is derived or handled by padding.
    const std::vector<float>& mean = {0.5f}, // Example for grayscale
    const std::vector<float>& stddev = {0.5f}, // Example for grayscale
    bool to_grayscale = true
);


// Helper to convert cv::Mat to CHW float vector - can be used by preprocess functions
std::vector<float> matToCHWVector(const cv::Mat& mat, const std::vector<float>& mean, const std::vector<float>& stddev);

} // namespace ImageUtils

#endif // IMAGE_UTILS_H
