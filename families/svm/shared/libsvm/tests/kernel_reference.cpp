// Validation only: execute the pinned sklearn LIBSVM training-kernel methods.
#define _DENSE_REP
#include "svm.cpp"
#include <iomanip>
#include <iostream>
#include <vector>
#include <stdexcept>

static double reference_dot(int n, const double* x, int sx,
                            const double* y, int sy) {
    double result = 0;
    for (int i = 0; i < n; ++i) result += x[i * sx] * y[i * sy];
    return result;
}

class TrainingKernel : public svm::Kernel {
public:
    using svm::Kernel::Kernel;
    double evaluate(int i, int j) const { return (this->*kernel_function)(i, j); }
    Qfloat* get_Q(int, int) const override { throw std::logic_error("unused"); }
    double* get_QD() const override { throw std::logic_error("unused"); }
};

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const int rows = std::stoi(argv[1]), cols = std::stoi(argv[2]);
    std::vector<double> values(rows * cols);
    std::vector<svm_node> nodes(rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            values[i * cols + j] = ((i * 17 + (2 * cols - 1 - 2 * j) * 3) % 31 - 15) / 97.0;
        nodes[i] = {cols, i, values.data() + i * cols};
    }
    svm_parameter parameters{};
    parameters.kernel_type = std::stoi(argv[3]);
    parameters.gamma = 0.7;
    parameters.degree = 3;
    parameters.coef0 = 0.2;
    BlasFunctions blas{reference_dot};
    TrainingKernel kernel(rows, nodes.data(), parameters, &blas);
    std::cout << std::setprecision(17);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < rows; ++j)
            std::cout << kernel.evaluate(i, j) << '\n';
}
