// Reading a weight file (docs/WEIGHTS_FORMAT.md): the key = value header and the float64 arrays
// that follow it. Internal to the library; NesModel::load uses it.
#pragma once

#include "nes_cpp/index.hpp"

#include <bit>
#include <charconv>
#include <istream>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace nes_cpp::detail {

// Weight files store float64 in little-endian order; reading them as raw bytes needs the same
// order in memory. x86-64 and ARM64 are little-endian.
static_assert(std::endian::native == std::endian::little, "weight files are little-endian");

// Reads a weight file header (key = value lines up to end_header) and gives typed access to it.
// Keys are removed as they are read, so that unknown keys can be reported at the end.
class WeightsHeader {
  public:
    WeightsHeader(std::istream& file, std::string path) : path_(std::move(path)) {
        std::string line;
        while (std::getline(file, line)) { // leaves `file` at the first weight byte
            if (line == "end_header") {
                return;
            }
            const std::size_t equals = line.find(" = ");
            if (equals == std::string::npos) {
                fail("bad header line '" + line + "'");
            }
            values_[line.substr(0, equals)] = line.substr(equals + 3);
        }
        fail("no end_header line");
    }

    [[noreturn]] void fail(const std::string& what) const {
        throw std::invalid_argument("NesModel: " + path_ + ": " + what);
    }

    std::string text(const std::string& key) {
        const auto it = values_.find(key);
        if (it == values_.end()) {
            fail("missing key '" + key + "'");
        }
        std::string value = it->second;
        values_.erase(it);
        return value;
    }

    // Space-separated numbers; std::from_chars reads the shortest round-trip form exactly and
    // does not depend on the locale.
    template <typename T> std::vector<T> numbers(const std::string& key) {
        const std::string value = text(key);
        std::vector<T> result;
        const char* p = value.data();
        const char* const end = p + value.size();
        while (p < end) {
            T x{};
            const auto [next, error] = std::from_chars(p, end, x);
            if (error != std::errc() || (next < end && *next != ' ')) {
                fail("bad number in '" + key + " = " + value + "'");
            }
            result.push_back(x);
            p = next < end ? next + 1 : next;
        }
        return result;
    }

    template <typename T> T number(const std::string& key) {
        const std::vector<T> values = numbers<T>(key);
        if (values.size() != 1) {
            fail("'" + key + "' needs one value");
        }
        return values[0];
    }

    bool flag(const std::string& key) {
        const int value = number<int>(key);
        if (value != 0 && value != 1) {
            fail("'" + key + "' must be 0 or 1");
        }
        return value == 1;
    }

    void check_all_read() const {
        if (!values_.empty()) {
            fail("unknown key '" + values_.begin()->first + "'");
        }
    }

  private:
    std::string path_;
    std::map<std::string, std::string> values_;
};

// Reads `count` float64 values that follow in the file.
inline std::vector<double> read_doubles(std::istream& file, Index count,
                                        const WeightsHeader& header) {
    std::vector<double> values(static_cast<std::size_t>(count));
    const auto bytes = static_cast<std::streamsize>(count * Index{sizeof(double)});
    file.read(reinterpret_cast<char*>(values.data()), bytes);
    if (file.gcount() != bytes) {
        header.fail("the file ends before all weights are read");
    }
    return values;
}

} // namespace nes_cpp::detail
