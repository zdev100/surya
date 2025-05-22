#ifndef JSON_UTILS_H
#define JSON_UTILS_H

#include <string>
#include <vector>
#include "layout_parser.h"   // For LayoutData structure
// #include "ocr_parser.h"   // OCR removed
#include <nlohmann/json.hpp> // nlohmann/json library

// Ensure LayoutData struct definition is fully available here.
// For cv::Point, opencv headers should be included by layout_parser.h

namespace JsonUtils {

/**
 * @brief Generates a nlohmann::json object from LayoutData, including image metadata.
 *
 * The JSON structure aims to be compatible with surya.detection.schema.TextDetectionResult
 * but will only contain layout information.
 *
 * @param layout_data The layout data containing detected polygons.
 * @param image_path Path to the original image.
 * @param image_width Original image width.
 * @param image_height Original image height.
 * @return A nlohmann::json object representing the layout data and image metadata.
 */
nlohmann::json layoutDataToJson(
    const LayoutData& layout_data,
    const std::string& image_path,
    int image_width,
    int image_height
);

// Removed ocrDataToJson
// Removed createComprehensiveJson (or it's merged into layoutDataToJson)

/**
 * @brief Saves a nlohmann::json object to a file.
 *
 * @param json_data The nlohmann::json object to save.
 * @param output_json_path The path to the file where the JSON should be saved.
 * @return True if saving was successful, false otherwise.
 */
bool saveJsonToFile(const nlohmann::json& json_data, const std::string& output_json_path);

} // namespace JsonUtils

#endif // JSON_UTILS_H
