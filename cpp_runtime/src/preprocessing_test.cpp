#include "preprocessing.hpp"
#include <iostream>
#include <fstream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <output_tensor.bin>" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    std::string output_path = argv[2];

    try {
        std::vector<float> tensor = nanosam::preprocessImage(image_path, 1024);

        std::ofstream out(output_path, std::ios::binary);
        if (!out) {
            std::cerr << "Failed to open output file: " << output_path << std::endl;
            return 1;
        }
        out.write(reinterpret_cast<const char*>(tensor.data()), tensor.size() * sizeof(float));
        out.close();

        std::cout << "Preprocessed tensor written: " << output_path
                  << " (" << tensor.size() * sizeof(float) << " bytes)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
