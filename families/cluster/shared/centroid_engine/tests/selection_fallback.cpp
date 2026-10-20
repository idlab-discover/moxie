// Test-only oracle for the standard-library fallback used by NumPy's argselect.
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

template<class T> void check(int argc, char** argv) {
    std::vector<T> values;
    for (int i = 2; i < argc; ++i) {
        T value = static_cast<T>(std::strtod(argv[i], nullptr));
        values.push_back(value * value);
    }
    std::vector<size_t> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return values[a] < values[b];
    });
    for (auto it = indices.rbegin(); it != indices.rend(); ++it)
        std::cout << *it << '\n';
}

int main(int argc, char** argv) {
    if (argc < 3) return 1;
    if (std::string(argv[1]) == "f32") check<float>(argc, argv);
    else check<double>(argc, argv);
}
