#ifndef VISUALIZATION_UTILS_H
#define VISUALIZATION_UTILS_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp> // OpenCV for drawing
#include "layout_parser.h"    // For LayoutData
#include "ocr_parser.h"       // For OCRData

// Ensure LayoutData and OCRData struct definitions are fully available.

namespace VisualizationUtils {

/**
 * @brief Draws layout polygons and OCR text onto an image and saves it.
 *
 * @param image The original image (cv::Mat) on which to draw.
 * @param layout_data The layout data containing detected polygons.
 * @param ocr_data The OCR data containing recognized text lines and their polygons.
 *                 This can be empty or have an empty lines vector if no OCR is performed.
 * @param output_image_path The path where the visualized image will be saved.
 * @return True if the image was saved successfully, false otherwise.
 */
bool drawLayoutAndOCR(
    const cv::Mat& image,
    const LayoutData& layout_data,
    const OCRData& ocr_data, // Pass by const reference
    const std::string& output_image_path
);

} // namespace VisualizationUtils

#endif // VISUALIZATION_UTILS_H
