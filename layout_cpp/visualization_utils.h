#ifndef VISUALIZATION_UTILS_H
#define VISUALIZATION_UTILS_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp> // OpenCV for drawing
#include "layout_parser.h"    // For LayoutData
// #include "ocr_parser.h"    // OCR removed

// Ensure LayoutData struct definition is fully available.

namespace VisualizationUtils {

/**
 * @brief Draws layout polygons onto an image and saves it.
 *
 * @param image The original image (cv::Mat) on which to draw.
 * @param layout_data The layout data containing detected polygons.
 * @param output_image_path The path where the visualized image will be saved.
 * @return True if the image was saved successfully, false otherwise.
 */
bool drawLayout(
    const cv::Mat& image,
    const LayoutData& layout_data,
    const std::string& output_image_path
);

} // namespace VisualizationUtils

#endif // VISUALIZATION_UTILS_H
