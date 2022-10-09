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

        class Tensor
        {
            int width = 0;
            int height = 0;
            int depth = 0;
            float* data;
        };
        // ===============================================
        // Array object.
        //
        // vector<vector<float>> tensor = 2d vector tensor.
        // int width = width of matrix.
        // int height = height of matrix.
        // ===============================================
        class Array
        {
            public:
                vector<vector<float>> array;
                int width = 0;
                int height = 0;
                float *tensor;

                // ===============================================
                // Tensor object constructors.
                // ===============================================
                Array();
                Array(vector<vector<float>> arr);

                // ===============================================
                // Tensor object functions.
                // ===============================================
                int setWidth();
                int setHeight();
                float* setTensor(vector<vector<float>> arr);
        };

        // ===============================================
        // Derived array object, creates tensor of (w, h) dimensions filled with zeros.
        // ===============================================
        class Zeros : public Array
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
        // Derived array object, creates tensor of (w, h) dimensions filled with ones.
        // ===============================================
        class Ones : public Array
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