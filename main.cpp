#include "LinearAlgebra.h"

int main(void) {

    Matrix<int> A(3, 3, 0);
    Matrix<int> B(3, 3, 0);

    A.elements = {  2, 3, 0, 
                    5, 0, 6,
                    1, 8, 7  };
    B.elements = {  8, 9, 1,
                    3, 0, 0,
                    2, 2, 8  };

    Matrix<int> C(3, 3, 0);
    
    C = A + B;

    std::cout << C.toString() << '\n' << std::endl;

    Matrix<int> v1(3, 1, 0);
    Matrix<int> v2(3, 1, 0);

    v1.elements = { 3, 2, -5 };
    v2.elements = { 2, -6, 4 };
    
    std::cout << "Outer Product:" << std::endl;
    Matrix<int> crossProductResult = outer(v1, v2);
    std::cout << crossProductResult.toString() << '\n' << std::endl;

    std::cout << "Dot Product:" << std::endl;
    i32 dotProductResult = dot(v1, v2);
    std::cout << dotProductResult << '\n' << std::endl;

    return 0;
}