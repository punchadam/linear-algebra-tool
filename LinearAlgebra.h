#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H

#include <stdint.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <concepts>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

// MatrixLike concept

// element reference helper
template <class M>
using elem_ref_t = decltype(std::declval<M&>()(std::size_t{}, std::size_t{}));

template <class M>
using elem_t = std::remove_cvref_t<elem_ref_t<M>>;

// a matrix-like object must provide rows(), columns(), and operator()(i,j)
template <class M>
concept MatrixLike =
requires(M m, const M cm, std::size_t i, std::size_t j) {
    { cm.rows() }    -> std::convertible_to<std::size_t>;
    { cm.columns() } -> std::convertible_to<std::size_t>;
    { cm(i, j) };
    { m(i, j) };
};

// Matrix

template <class T>
struct Matrix {
private:
    size_t _m = 0;
    size_t _n = 0;

public:
    std::vector<T> elements;

    Matrix() = default;

    // initializes and fills an m x n matrix
    Matrix(size_t m, size_t n, const T& fill = T{})
        : _m(m), _n(n), elements(m * n, fill) {}

    size_t rows() const { return _m; }
    size_t columns() const { return _n; }
    size_t size() const { return elements.size(); }

    // overloads '()' so i can access element (i, j) of A by just doing A(i, j)
    T& operator()(size_t i, size_t j) { return elements[i * _n + j]; }
    const T& operator()(size_t i, size_t j) const { return elements[i * _n + j]; }

    void resize(size_t newM, size_t newN, const T& fill = T{}) {
        _m = newM;
        _n = newN;
        elements.assign(_m * _n, fill);
    }

    Matrix operator-() const {
        Matrix out(_m, _n);
        for (size_t i = 0; i < elements.size(); i++) {
            out.elements[i] = -elements[i];
        }
        return out;
    }

    std::string toString() const {
        std::stringstream buffer;
        for (size_t i = 0; i < _m; i++) {
            buffer << "\n[ ";
            for (size_t j = 0; j < _n; j++) {
                buffer << " " << (*this)(i, j);
                if (j < _n - 1) buffer << ",";
            }
            buffer << " ]\n";
        }
        buffer << "\n";
        return buffer.str();
    }
};

// TransposeView

// read-only transpose
template <MatrixLike M>
struct TransposeView {
    const M& a;

    size_t rows() const { return a.columns(); }
    size_t columns() const { return a.rows(); }

    decltype(auto) operator()(size_t i, size_t j) const {
        return a(j, i);
    }
};

template <MatrixLike M>
TransposeView<M> transpose(const M& m) { return { m }; }

// Vector helpers

template <MatrixLike M>
bool isVector(const M& v) {
    return v.rows() == 1 || v.columns() == 1;
}

template <MatrixLike M>
size_t vectorLength(const M& v) {
    return (v.rows() == 1) ? v.columns() : v.rows();
}

template <MatrixLike M>
decltype(auto) vectorAt(const M& v, const size_t i) {
    return (v.rows() == 1) ? v(0, i) : v(i, 0);
}

template <MatrixLike M>
std::string toString(const M& m) {
    std::stringstream buffer;

    for (size_t i = 0; i < m.rows(); i++) {
        buffer << "[ ";
        for (size_t j = 0; j < m.columns(); j++) {
            buffer << m(i, j);
            if (j < m.columns() - 1) buffer << ", ";
        }
        buffer << " ]\n";
    }

    return buffer.str();
}

// Math Operations

// two template types are used for math to account for
// matrix-like objects like TransposeView

// dot product
template <MatrixLike A, MatrixLike B>
requires requires(const A& a, const B& b) { vectorAt(a,0) * vectorAt(b,0); }
auto dot(const A& a, const B& b) {

    if (!isVector(a) || !isVector(b)) {
        throw std::runtime_error("Dot Product: Args must be (m x 1) or (1 x n) vectors");
    }

    if (vectorLength(a) != vectorLength(b)) {
        throw std::runtime_error("Dot Product: Vectors must have the same length");
    }

    using T = std::remove_cvref_t<decltype(vectorAt(a,0) * vectorAt(b,0))>;
    T sum{};

    for (size_t i = 0; i < vectorLength(a); i++) {
        sum += vectorAt(a, i) * vectorAt(b, i);
    }

    return sum;
}

template <MatrixLike A, MatrixLike B>
requires requires(const A& a, const B& b) { vectorAt(a,0) * vectorAt(b,0); }
auto outer(const A& a, const B& b) {

    if (!isVector(a) || !isVector(b)) {
        throw std::runtime_error("Outer Product: Args must be (m x 1) or (1 x n) vectors");
    }

    using T = std::remove_cvref_t<decltype(vectorAt(a,0) * vectorAt(b,0))>;

    size_t m = vectorLength(a);
    size_t n = vectorLength(b);

    Matrix<T> out(m, n, T{});

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            out(i, j) = vectorAt(a, i) * vectorAt(b, j);
        }
    }

    return out;
}

// matrix multiplication
template <MatrixLike A, MatrixLike B>
requires requires(const A& a, const B& b) { a(0,0) * b(0,0); }
auto matrixMultiply(const A& a, const B& b) {

    if (a.columns() != b.rows()) {
        throw std::runtime_error("Matrix Multiplication: Dimension mismatch.");
    }

    using T = std::remove_cvref_t<decltype(a(0,0) * b(0,0))>;

    Matrix<T> c(a.rows(), b.columns(), T{});

    // for each row i in a
    for (size_t i = 0; i < a.rows(); i++) {

        // for each column j in b
        for (size_t j = 0; j < b.columns(); j++) {

            T sum{};

            // calculate the dot product of a's row i and b's column j
            for (size_t k = 0; k < a.columns(); k++) {
                sum += a(i, k) * b(k, j);
            }

            // that dot product becomes c(i, j)
            c(i, j) = sum;
        }
    }

    return c;
}

template <MatrixLike A, MatrixLike B>
auto operator*(const A& a, const B& b) {
    return matrixMultiply(a, b);
}


// matrix addition
template <MatrixLike A, MatrixLike B>
requires requires(const A& a, const B& b) { a(0,0) + b(0,0); }
auto operator+(const A& a, const B& b) {

    if (a.columns() != b.columns() || a.rows() != b.rows()) {
        throw std::runtime_error("Matrix Addition: Dimension mismatch.");
    }

    using T = std::remove_cvref_t<decltype(a(0,0) + b(0,0))>;
    Matrix<T> c(a.rows(), a.columns(), T{});

    for (size_t i = 0; i < a.rows(); i++) {
        for (size_t j = 0; j < a.columns(); j++) {
            c(i, j) = a(i, j) + b(i, j);
        }
    }

    return c;
}

// matrix subtraction
template <MatrixLike A, MatrixLike B>
requires requires(const A& a, const B& b) { a(0,0) - b(0,0); }
auto operator-(const A& a, const B& b) {

    if (a.columns() != b.columns() || a.rows() != b.rows()) {
        throw std::runtime_error("Matrix Subtraction: Dimension mismatch.");
    }

    using T = std::remove_cvref_t<decltype(a(0,0) - b(0,0))>;
    Matrix<T> c(a.rows(), a.columns(), T{});

    for (size_t i = 0; i < a.rows(); i++) {
        for (size_t j = 0; j < a.columns(); j++) {
            c(i, j) = a(i, j) - b(i, j);
        }
    }

    return c;
}

#endif