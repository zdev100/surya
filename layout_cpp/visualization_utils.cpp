#include "visualization_utils.h"
#include <opencv2/imgproc.hpp> // For cv::polylines, cv::putText
#include <opencv2/imgcodecs.hpp> // For cv::imwrite
#include <iostream> // For std::cerr

// Ensure LayoutData struct definition is fully available.

namespace VisualizationUtils {

bool drawLayout( // Function name changed
    const cv::Mat& image,
    const LayoutData& layout_data,
    const std::string& output_image_path) {

    if (image.empty()) {
        std::cerr << "Error: Input image for visualization is empty." << std::endl;
        return false;
    }

    cv::Mat visualized_image = image.clone();

    // Draw layout polygons
    for (const auto& polygon_points : layout_data.detected_polygons) {
        if (polygon_points.size() > 1) { 
            const cv::Point* ppt[1] = { polygon_points.data() };
            int npt[] = { static_cast<int>(polygon_points.size()) };
            cv::polylines(visualized_image, ppt, npt, 1, true, cv::Scalar(0, 255, 0), 2); // Green lines, thickness 2
        }
    }

    // OCR text drawing removed

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
