#include "json_utils.h"
#include <fstream>      // For std::ofstream
#include <iostream>     // For std::cerr

// Ensure LayoutData definition is available
// This should come from including layout_parser.h
// For cv::Point, opencv headers should be included by layout_parser.h

namespace JsonUtils {

nlohmann::json layoutDataToJson(
    const LayoutData& layout_data,
    const std::string& image_path,
    int image_width,
    int image_height) {

    nlohmann::json j;
    j["image_path"] = image_path;
    j["image_width"] = image_width;
    j["image_height"] = image_height;
    j["version"] = "surya_layout_cpp_v0.1_alpha"; // Updated version string

    j["layout_polygons"] = nlohmann::json::array(); 

    for (const auto& polygon_points : layout_data.detected_polygons) {
        nlohmann::json current_polygon = nlohmann::json::array();
        for (const auto& point : polygon_points) {
            current_polygon.push_back({point.x, point.y});
        }
        j["layout_polygons"].push_back(current_polygon);
    }
    
    // Include raw_output_info if it's not empty and potentially useful
    if (!layout_data.raw_output_info.empty()) {
        j["raw_model_output_info"] = layout_data.raw_output_info;
    }

    return j;
}

// ocrDataToJson removed
// createComprehensiveJson removed (functionality merged into layoutDataToJson)

bool saveJsonToFile(const nlohmann::json& json_data, const std::string& output_json_path) {
    std::ofstream o(output_json_path);
    if (!o.is_open()) {
        std::cerr << "Error: Could not open JSON file for writing: " << output_json_path << std::endl;
        return false;
    }
    try {
        o << json_data.dump(4); // pretty print with 4 spaces
        o.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error writing JSON to file: " << e.what() << std::endl;
        // Attempt to close file even if write failed, though it might already be closed or in bad state
        if (o.is_open()) {
            o.close();
        }
        return false;
    }
}

} // namespace JsonUtils
