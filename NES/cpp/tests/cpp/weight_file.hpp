// Test helper: a weight file (docs/WEIGHTS_FORMAT.md) written by hand into the temporary folder.
#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

// The file is deleted when the object goes out of scope (RAII).
struct WeightFile {
    std::filesystem::path path;

    WeightFile(const std::string& name, const std::vector<std::string>& header,
               const std::vector<double>& weights)
        : path(std::filesystem::temp_directory_path() / ("nes_cpp_test_" + name + ".nesw")) {
        std::ofstream file(path, std::ios::binary);
        for (const std::string& line : header) {
            file << line << '\n';
        }
        file << "end_header\n";
        file.write(reinterpret_cast<const char*>(weights.data()),
                   static_cast<std::streamsize>(weights.size() * sizeof(double)));
    }
    ~WeightFile() {
        std::error_code ignored; // never throw from a destructor
        std::filesystem::remove(path, ignored);
    }
    WeightFile(const WeightFile&) = delete; // one owner per file
    WeightFile& operator=(const WeightFile&) = delete;
};
