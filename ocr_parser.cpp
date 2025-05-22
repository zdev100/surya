#include "ocr_parser.h"
#include "image_utils.h"    // For preprocessing image patches
#include "onnx_inference.h" // For tensor creation and running inference (though session is passed)
#include <stdexcept>        // For std::runtime_error
#include <iostream>         // For std::cout, std::cerr for logging

// Need the full definition of LayoutData if we are accessing its members like detected_polygons
// This should ideally be in a common header file or layout_parser.h should be included.
// For now, ensure it's compatible with the definition in main.cpp or layout_parser.h
struct LayoutData { // Re-affirming structure for this cpp; ideally from a .h
    std::vector<std::vector<cv::Point>> detected_polygons;
    std::string raw_output_info;
};


OCRData processOCR(
    const cv::Mat& original_image,
    const LayoutData& layout_results,
    Ort::Session& recognition_session,
    Ort::AllocatorWithDefaultOptions& allocator,
    const char* input_node_name,
    const std::vector<const char*>& output_node_names,
    bool verbose // Added verbose flag
) {
    if (verbose) std::cout << "processOCR: Starting OCR processing for " << layout_results.detected_polygons.size() << " regions." << std::endl;

    if (original_image.empty()) {
        if (verbose) std::cerr << "processOCR Error: Input original_image is empty." << std::endl;
        throw std::runtime_error("Input original_image to processOCR is empty.");
    }
    if (layout_results.detected_polygons.empty()) {
        if (verbose) std::cout << "processOCR: No layout polygons provided. Skipping OCR." << std::endl;
        return {}; // Return empty OCRData
    }
    if (!input_node_name || output_node_names.empty()) {
        if (verbose) std::cerr << "processOCR Error: Input or output node names are not specified for OCR model." << std::endl;
        throw std::runtime_error("Input or output node names are not specified for OCR model.");
    }

    OCRData all_ocr_results;
    int region_idx = 0;

    // --- For each detected polygon from layout analysis ---
    for (const auto& polygon : layout_results.detected_polygons) {
        region_idx++;
        if (verbose) std::cout << "processOCR: Processing region " << region_idx << "/" << layout_results.detected_polygons.size() << std::endl;

        if (polygon.size() < 3) { 
            if (verbose) std::cerr << "processOCR Warning: Skipping invalid polygon with less than 3 points for region " << region_idx << "." << std::endl;
            continue;
        }

        // --- 1. Crop Image Patch ---
        cv::Rect bounding_rect = cv::boundingRect(polygon);
        bounding_rect &= cv::Rect(0, 0, original_image.cols, original_image.rows);
        if (bounding_rect.width <= 0 || bounding_rect.height <= 0) {
            if (verbose) std::cerr << "processOCR Warning: Skipping empty or invalid bounding rectangle for region " << region_idx << "." << std::endl;
            continue;
        }
        cv::Mat image_patch = original_image(bounding_rect);
        if (verbose) std::cout << "processOCR: Cropped patch for region " << region_idx << " (x,y,w,h): " << bounding_rect.x << "," << bounding_rect.y << "," << bounding_rect.width << "," << bounding_rect.height << std::endl;


        // --- 2. Preprocess the Image Patch for OCR Model ---
        // These parameters should match the OCR model's requirements.
        int model_target_ocr_height = 64; 
        std::vector<float> ocr_mean = {0.5f};
        std::vector<float> ocr_stddev = {0.5f};
        bool ocr_to_grayscale = true;

        std::vector<float> preprocessed_patch_data;
        try {
            preprocessed_patch_data = ImageUtils::preprocessOCRImage(
                image_patch, model_target_ocr_height, ocr_mean, ocr_stddev, ocr_to_grayscale);
        } catch (const std::exception& e) {
            if (verbose) std::cerr << "processOCR Error: Exception during OCR image patch preprocessing for region " << region_idx << ": " << e.what() << std::endl;
            continue; 
        }
        if (preprocessed_patch_data.empty()){
            if (verbose) std::cerr << "processOCR Warning: Preprocessed data is empty for region " << region_idx << ", skipping." << std::endl;
            continue;
        }
        if (verbose) std::cout << "processOCR: Preprocessing complete for patch " << region_idx << ". Data size: " << preprocessed_patch_data.size() << std::endl;


        // --- 3. Create Input Tensor for OCR ---
        int patch_channels = ocr_to_grayscale ? 1 : image_patch.channels();
        int64_t patch_width = 0;
        if (patch_channels > 0 && model_target_ocr_height > 0 ) { // preprocessed_patch_data.empty() already checked
             patch_width = preprocessed_patch_data.size() / (patch_channels * model_target_ocr_height);
        } else { // Should not happen if preprocessed_patch_data is not empty
            if (verbose) std::cerr << "processOCR Warning: Cannot determine patch_width for region " << region_idx << " (channels: " << patch_channels << ", height: " << model_target_ocr_height << "). Skipping." << std::endl;
            continue;
        }
        if (patch_width <=0){
            if (verbose) std::cerr << "processOCR Warning: Calculated patch_width is " << patch_width << " for region " << region_idx << ". Skipping." << std::endl;
            continue;
        }


        std::vector<int64_t> ocr_input_shape = {1, static_cast<int64_t>(patch_channels), model_target_ocr_height, patch_width};
        if (verbose) std::cout << "processOCR: Input tensor shape for region " << region_idx << ": [1, " << patch_channels << ", " << model_target_ocr_height << ", " << patch_width << "]" << std::endl;

        Ort::Value ocr_input_tensor = Ort::Value::CreateTensor<float>(
            allocator.GetInfo(), preprocessed_patch_data.data(), preprocessed_patch_data.size(),
            ocr_input_shape.data(), ocr_input_shape.size());

        // --- 4. Run OCR Inference ---
        if (verbose) std::cout << "processOCR: Running inference for region " << region_idx << "..." << std::endl;
        std::vector<Ort::Value> ocr_output_tensors;
        try {
            ocr_output_tensors = recognition_session.Run(
                Ort::RunOptions{nullptr},
                &input_node_name, &ocr_input_tensor, 1,
                output_node_names.data(), output_node_names.size()
            );
        } catch (const Ort::Exception& e) {
            if (verbose) std::cerr << "processOCR Error: Exception during OCR model inference for region " << region_idx << ": " << e.what() << std::endl;
            continue; 
        }
        if (verbose) std::cout << "processOCR: Inference completed for region " << region_idx << ". Received " << ocr_output_tensors.size() << " output tensor(s)." << std::endl;

        // --- 5. Process OCR Output (Simplified Placeholder Parsing) ---
        OCRData::TextLine current_line;
        current_line.polygon = polygon; 
        current_line.confidence = 0.90f + (static_cast<float>(rand() % 10) / 100.0f); // Random confidence 0.90-0.99

        if (!ocr_output_tensors.empty()) {
            Ort::Value& first_ocr_output = ocr_output_tensors[0];
            Ort::TensorTypeAndShapeInfo ocr_tensor_info = first_ocr_output.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> ocr_output_shape = ocr_tensor_info.GetShape();

            std::string temp_info;
            if (verbose) {
                temp_info = "OCR model output for patch " + std::to_string(region_idx) + " shape: [";
                for (size_t i = 0; i < ocr_output_shape.size(); ++i) {
                    temp_info += std::to_string(ocr_output_shape[i]) + (i == ocr_output_shape.size() - 1 ? "" : ", ");
                }
                temp_info += "]";
                std::cout << "processOCR: " << temp_info << std::endl;
                all_ocr_results.raw_output_info += temp_info + "; ";
            }
            
            // Placeholder decoding:
            // Assume output is a sequence of token IDs, e.g., shape [Batch, SeqLen] or [Batch, SeqLen, VocabSize]
            // For simplicity, generate text based on tensor dimensions or a fixed string.
            if (ocr_output_shape.size() >= 2) { // e.g. [1, SeqLen] or [1, SeqLen, Prob]
                // const int64_t* token_ids = first_ocr_output.GetTensorData<int64_t>(); // If output is int64 IDs
                // const float* probs = first_ocr_output.GetTensorData<float>(); // If output is probabilities
                // int sequence_length = ocr_output_shape[1];
                // Simple placeholder:
                current_line.text = "TxtRgn" + std::to_string(region_idx);
                // Example of very basic "decoding" if token IDs were available:
                // for(int i=0; i < sequence_length; ++i) {
                //    int token_id = token_ids[i]; // This is hypothetical
                //    if (token_id % 3 == 0) current_line.text += "a";
                //    else if (token_id % 3 == 1) current_line.text += "b";
                //    else current_line.text += "c";
                // }
            } else {
                 current_line.text = "UndecodedRgn" + std::to_string(region_idx);
            }
             if (verbose) std::cout << "processOCR: Placeholder text for region " << region_idx << ": \"" << current_line.text << "\"" << std::endl;

        } else {
            current_line.text = "OCR_NoOutput_Rgn" + std::to_string(region_idx);
            if (verbose) std::cerr << "processOCR: No output tensors for region " << region_idx << "." << std::endl;
            all_ocr_results.raw_output_info += "OCR model produced no output for patch " + std::to_string(region_idx) + "; ";
        }
        all_ocr_results.lines.push_back(current_line);
    }

    if (verbose) std::cout << "processOCR: OCR processing finished for all regions. Total lines: " << all_ocr_results.lines.size() << std::endl;
    return all_ocr_results;
}
