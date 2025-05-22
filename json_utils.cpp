#include "json_utils.h"
#include <fstream>      // For std::ofstream
#include <iostream>     // For std::cerr

// Ensure LayoutData and OCRData definitions are available
// These should come from including layout_parser.h and ocr_parser.h
// For cv::Point, opencv headers should be included by layout_parser.h or ocr_parser.h

namespace JsonUtils {

nlohmann::json layoutDataToJson(const LayoutData& layout_data) {
    nlohmann::json j;
    j["bboxes"] = nlohmann::json::array(); // Mimicking surya.detection.schema.TextDetectionResult structure

    for (const auto& polygon_points : layout_data.detected_polygons) {
        nlohmann::json current_polygon = nlohmann::json::array();
        for (const auto& point : polygon_points) {
            current_polygon.push_back({point.x, point.y});
        }
        j["bboxes"].push_back(current_polygon);
    }
    // Other fields from TextDetectionResult like "image_path", "image_height", "image_width"
    // would be added in the comprehensive JSON creation step or if needed directly here.
    return j;
}

nlohmann::json ocrDataToJson(const OCRData& ocr_data) {
    nlohmann::json j_lines = nlohmann::json::array();

    for (const auto& line : ocr_data.lines) {
        nlohmann::json j_line;
        j_line["text"] = line.text;

        nlohmann::json current_polygon = nlohmann::json::array();
        for (const auto& point : line.polygon) {
            current_polygon.push_back({point.x, point.y});
        }
        j_line["polygon"] = current_polygon;
        j_line["confidence"] = line.confidence; // Add confidence if available
        j_lines.push_back(j_line);
    }
    return j_lines; // Returns an array of text line objects
}

nlohmann::json createComprehensiveJson(
    const std::string& image_path,
    int image_width,
    int image_height,
    const nlohmann::json& layout_json, // Result from layoutDataToJson
    const nlohmann::json& ocr_json     // Result from ocrDataToJson (an array of lines)
) {
    nlohmann::json comprehensive_json;
    comprehensive_json["image_path"] = image_path;
    comprehensive_json["image_width"] = image_width;
    comprehensive_json["image_height"] = image_height;

    // Embed the bboxes array from layout_json directly
    if (layout_json.contains("bboxes")) {
        comprehensive_json["layout_polygons"] = layout_json["bboxes"];
    } else {
        comprehensive_json["layout_polygons"] = nlohmann::json::array(); // Empty if no bboxes
    }

    // Embed the OCR lines array directly
    comprehensive_json["ocr_lines"] = ocr_json; // ocr_json is already an array of line objects

    // Add any other top-level information if needed
    // For example, language, processing time etc. (not part of this subtask yet)
    comprehensive_json["version"] = "surya_cpp_v0.1_alpha";

    return comprehensive_json;
}


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
