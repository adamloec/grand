// ===============================================
// Author: Adam Loeckle
// Date: 10/3/2022
// Description: Tensor object header file.
// ===============================================

#ifndef CORE_INCL
#define CORE_INCL
    #include <stdio.h>
    #include <iostream>
    #include <vector>
    using namespace std;
#endif

namespace Grand
{
    namespace Tensor
    {
        // ===============================================
        // Tensor object.
        //
        // vector<vector<float>> tensor = 2d vector tensor.
        // int width = width of matrix.
        // int height = height of matrix.
        // ===============================================
        class Matrix
        {
            public:
                vector<vector<float>> matrix;
                int width = 0;
                int height = 0;
                float *tensor;

                // ===============================================
                // Tensor object constructors.
                // ===============================================
                Matrix();
                Matrix(vector<vector<float>> matrix);

                // ===============================================
                // Tensor object functions.
                // ===============================================
                int setWidth();
                int setHeight();
                float* setTensor(vector<vector<float>> mat);
        };

        // ===============================================
        // Derived tensor object, creates tensor of (w, h) dimensions filled with zeros.
        // ===============================================
        class Zeros : public Matrix
        {
            public:
                // ===============================================
                // Zero tensor object constructors.
                // ===============================================
                Zeros();

                // ===============================================
                // Input (width, height) tensor constructor.
                //
                // int w = Desired width of tensor.
                // int h = Desired height of tensor.
                // ===============================================
                Zeros(int w, int h);
        };

        // ===============================================
        // Derived tensor object, creates tensor of (w, h) dimensions filled with ones.
        // ===============================================
        class Ones : public Matrix
        {
            public:
                // ===============================================
                // Default tensor constructor, nothing.
                // ===============================================
                Ones();

                // ===============================================
                // Input (width, height) tensor constructor.
                //
                // int w = Desired width of tensor.
                // int h = Desired height of tensor.
                // ===============================================
                Ones(int w, int h);
        };
    }
}