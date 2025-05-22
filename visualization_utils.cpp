#include "visualization_utils.h"
#include <opencv2/imgproc.hpp> // For cv::polylines, cv::putText
#include <opencv2/imgcodecs.hpp> // For cv::imwrite
#include <iostream> // For std::cerr

// Ensure LayoutData and OCRData struct definitions are fully available.

namespace VisualizationUtils {

bool drawLayoutAndOCR(
    const cv::Mat& image,
    const LayoutData& layout_data,
    const OCRData& ocr_data,
    const std::string& output_image_path) {

    if (image.empty()) {
        std::cerr << "Error: Input image for visualization is empty." << std::endl;
        return false;
    }

    cv::Mat visualized_image = image.clone();

    // Draw layout polygons
    for (const auto& polygon_points : layout_data.detected_polygons) {
        if (polygon_points.size() > 1) { // Need at least 2 points for a line, typically 3+ for a polygon
            // cv::polylines requires a C-style array of cv::Point arrays
            const cv::Point* ppt[1] = { polygon_points.data() };
            int npt[] = { static_cast<int>(polygon_points.size()) };
            cv::polylines(visualized_image, ppt, npt, 1, true, cv::Scalar(0, 255, 0), 2); // Green lines, thickness 2
        }
    }

    // Draw OCR text
    // This is a basic implementation. More sophisticated placement might be needed.
    for (const auto& text_line : ocr_data.lines) {
        if (!text_line.polygon.empty() && !text_line.text.empty()) {
            // Use the first point of the polygon as an anchor for the text
            // Or, calculate centroid or top-left of the bounding box of the polygon
            cv::Point text_origin = text_line.polygon[0];

            // Adjust origin slightly to be above the line or inside
            text_origin.y -= 5; // Move text slightly above the top point
            if (text_origin.y < 10) text_origin.y = text_line.polygon[0].y + 15; // If it goes off screen, put it below
            if (text_origin.x < 0) text_origin.x = 0;
            if (text_origin.x >= visualized_image.cols) text_origin.x = visualized_image.cols - 100; // Avoid going off screen

            cv::putText(
                visualized_image,
                text_line.text,
                text_origin,
                cv::FONT_HERSHEY_SIMPLEX,
                0.5, // Font scale
                cv::Scalar(255, 0, 0), // Blue color for text
                1,     // Thickness
                cv::LINE_AA
            );
        }
    }

    // Save the image
    try {
        bool success = cv::imwrite(output_image_path, visualized_image);
        if (!success) {
            std::cerr << "Error: Failed to save visualization image to: " << output_image_path << std::endl;
            return false;
        }
    } catch (const cv::Exception& ex) {
        std::cerr << "OpenCV exception while saving image: " << ex.what() << std::endl;
        return false;
    }

    std::cout << "Visualization saved to: " << output_image_path << std::endl;
    return true;
}

} // namespace VisualizationUtils
