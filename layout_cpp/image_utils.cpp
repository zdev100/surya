#include "image_utils.h"
#include <stdexcept> // For std::runtime_error
#include <iostream>  // For std::cout, std::cerr (debugging, can be removed later)

// OpenCV specific includes, already in image_utils.h via opencv.hpp but good for clarity
#include <opencv2/imgcodecs.hpp> // For cv::imread
#include <opencv2/imgproc.hpp>   // For cv::resize, cv::cvtColor, etc.

namespace ImageUtils {

cv::Mat loadImage(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        // Consider logging an error or throwing an exception
        std::cerr << "Error: Could not load image from path: " << image_path << std::endl;
    }
    return image;
}

// Helper function to convert cv::Mat to a flat float vector in CHW format
// and apply normalization.
std::vector<float> matToCHWVector(const cv::Mat& mat, const std::vector<float>& mean, const std::vector<float>& stddev) {
    if (mat.empty()) {
        throw std::runtime_error("Input cv::Mat is empty in matToCHWVector.");
    }
    if (mat.channels() != mean.size() || mat.channels() != stddev.size()) {
        throw std::runtime_error("Mismatch between Mat channels and mean/stddev size.");
    }

    cv::Mat float_mat;
    mat.convertTo(float_mat, CV_32F); // Convert to float

    std::vector<float> data(mat.channels() * mat.rows * mat.cols);
    float* data_ptr = data.data();

    for (int c = 0; c < mat.channels(); ++c) {
        for (int h = 0; h < mat.rows; ++h) {
            for (int w = 0; w < mat.cols; ++w) {
                float val = float_mat.at<cv::Vec3f>(h, w)[c]; // Assuming 3 channels for Vec3f
                if (mat.channels() == 1) { // Grayscale case
                     val = float_mat.at<float>(h,w);
                } else {
                     val = float_mat.at<cv::Vec<float,3>>(h,w)[c]; // General case for N channels
                }
                data_ptr[c * (mat.rows * mat.cols) + h * mat.cols + w] = (val / 255.0f - mean[c]) / stddev[c];
            }
        }
    }
    return data;
}


std::vector<float> preprocessLayoutImage(
    const cv::Mat& image,
    int target_height,
    int target_width,
    const std::vector<float>& mean,
    const std::vector<float>& stddev) {

    if (image.empty()) {
        throw std::runtime_error("Input image is empty for layout preprocessing.");
    }

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(target_width, target_height));

    // Convert BGR to RGB if the model expects RGB (OpenCV loads as BGR by default)
    // Most models trained with PyTorch expect RGB.
    cv::Mat rgb_image;
    cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);

    // Normalize and convert to CHW format
    // The mean and stddev are typically for RGB format.
    return matToCHWVector(rgb_image, mean, stddev);
}


std::vector<float> preprocessOCRImage(
    const cv::Mat& image_patch,
    int target_height,
    // int target_width, // OCR models might have fixed height, variable width
    const std::vector<float>& mean, // Often {0.5f} for grayscale if normalized to [-1, 1]
    const std::vector<float>& stddev, // Often {0.5f}
    bool to_grayscale) {

    if (image_patch.empty()) {
        throw std::runtime_error("Input image patch is empty for OCR preprocessing.");
    }

    cv::Mat processed_patch = image_patch.clone();

    if (to_grayscale && processed_patch.channels() == 3) {
        cv::cvtColor(processed_patch, processed_patch, cv::COLOR_BGR2GRAY);
    } else if (to_grayscale && processed_patch.channels() == 4) {
        cv::cvtColor(processed_patch, processed_patch, cv::COLOR_BGRA2GRAY);
    }

    // Resize to target height, maintaining aspect ratio for width (or padding if necessary)
    // This is a common strategy for OCR.
    int original_height = processed_patch.rows;
    int original_width = processed_patch.cols;
    double aspect_ratio = static_cast<double>(original_width) / static_cast<double>(original_height);
    int target_ocr_width = static_cast<int>(target_height * aspect_ratio);

    cv::Mat resized_patch;
    cv::resize(processed_patch, resized_patch, cv::Size(target_ocr_width, target_height));

    // Normalize and convert to CHW (even for grayscale, it might be 1xHxW)
    // For grayscale, mean and stddev should be single-element vectors.
    return matToCHWVector(resized_patch, mean, stddev);
}

} // namespace ImageUtils
