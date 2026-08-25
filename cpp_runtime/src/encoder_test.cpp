#include "trt_engine.hpp"
#include <iostream>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <encoder.engine> <input_tensor.bin> <output_embedding.bin>" << std::endl;
        return 1;
    }

    std::string encoder_path = argv[1];
    std::string input_path = argv[2];
    std::string output_path = argv[3];

    try {
        // Load input tensor [1,3,1024,1024] FP32
        std::ifstream in(input_path, std::ios::binary | std::ios::ate);
        if (!in) {
            std::cerr << "Failed to open input tensor: " << input_path << std::endl;
            return 1;
        }
        size_t size_bytes = in.tellg();
        in.seekg(0);
        size_t expected = 1 * 3 * 1024 * 1024 * sizeof(float);
        if (size_bytes != expected) {
            std::cerr << "Input tensor size mismatch: " << size_bytes << " bytes, expected " << expected << std::endl;
            return 1;
        }
        std::vector<float> input_tensor(1 * 3 * 1024 * 1024);
        in.read(reinterpret_cast<char*>(input_tensor.data()), size_bytes);
        in.close();

        // Load encoder
        nanosam::TrtEngine encoder;
        if (!encoder.loadEngine(encoder_path)) {
            std::cerr << "Failed to load encoder: " << encoder_path << std::endl;
            return 1;
        }
        encoder.allocateBuffers();

        // Copy input
        nanosam::DeviceBuffer* img_buf = encoder.getBuffer("image");
        nanosam::DeviceBuffer* emb_buf = encoder.getBuffer("image_embeddings");
        if (!img_buf || !emb_buf) {
            std::cerr << "Encoder tensors 'image' / 'image_embeddings' not found" << std::endl;
            return 1;
        }
        img_buf->copyFromHost(input_tensor.data(), input_tensor.size() * sizeof(float));

        // Infer
        if (!encoder.infer()) {
            std::cerr << "Encoder inference failed" << std::endl;
            return 1;
        }

        // Copy output
        std::vector<float> embeddings(256 * 64 * 64);
        emb_buf->copyToHost(embeddings.data(), embeddings.size() * sizeof(float));

        // Save
        std::ofstream out(output_path, std::ios::binary);
        if (!out) {
            std::cerr << "Failed to open output file: " << output_path << std::endl;
            return 1;
        }
        out.write(reinterpret_cast<const char*>(embeddings.data()), embeddings.size() * sizeof(float));
        out.close();

        std::cout << "Encoder embedding written: " << output_path
                  << " (" << embeddings.size() * sizeof(float) << " bytes)" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
