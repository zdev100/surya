#ifndef JSON_UTILS_H
#define JSON_UTILS_H

#include <string>
#include <vector>
#include "layout_parser.h"   // For LayoutData structure
#include "ocr_parser.h"      // For OCRData structure
#include <nlohmann/json.hpp> // nlohmann/json library

// Ensure LayoutData and OCRData struct definitions are fully available here.
// If they are defined in their respective .h files and those are included, it's fine.
// For cv::Point, opencv headers should be included by layout_parser.h or ocr_parser.h

namespace JsonUtils {

/**
 * @brief Generates a nlohmann::json object from LayoutData.
 *
 * The JSON structure aims to be compatible with surya.detection.schema.TextDetectionResult.
 * It will contain a list of polygons, where each polygon is a list of [x, y] coordinates.
 *
 * @param layout_data The layout data containing detected polygons.
 * @return A nlohmann::json object representing the layout data.
 */
nlohmann::json layoutDataToJson(const LayoutData& layout_data);


/**
 * @brief Generates a nlohmann::json object from OCRData, including layout polygons.
 *        This structure will be a list of objects, each containing the text,
 *        its polygon, and confidence (if available).
 *
 * @param ocr_data The OCR data containing recognized text lines and their polygons.
 * @return A nlohmann::json object representing the OCR data.
 */
nlohmann::json ocrDataToJson(const OCRData& ocr_data);


/**
 * @brief Creates a comprehensive JSON output combining image metadata, layout, and OCR results.
 *
 * @param image_path Path to the original image.
 * @param image_width Original image width.
 * @param image_height Original image height.
 * @param layout_json JSON object from layoutDataToJson.
 * @param ocr_json JSON object from ocrDataToJson (can be empty if no OCR).
 * @return nlohmann::json object for the final combined output.
 */
nlohmann::json createComprehensiveJson(
    const std::string& image_path,
    int image_width,
    int image_height,
    const nlohmann::json& layout_json, // Result from layoutDataToJson
    const nlohmann::json& ocr_json     // Result from ocrDataToJson
);


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
