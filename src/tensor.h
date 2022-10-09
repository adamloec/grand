// ===================================================================================================
// Author: Adam Loeckle
// Date: 10/3/2022
// Description: Tensor object header file.
// ===================================================================================================

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
        // Row = height (x)
        // Column = width (y)
        // Depth = depth (z)

        // ===================================================================================================
        // Tensor object, for kernel use. Instantiated inside of array objects and their derivitives.
        //
        // NOTE: This object should not be called outside of Array objects/derivatives.
        //
        // int width = width of tensor.
        // int height = height of tensor.
        // int depth = depth of tensor.
        // float* data = data of tensor.
        // ===================================================================================================
        class Tensor
        {
            public:
                int width = 0;
                int height = 0;
                int depth = 0;
                float* data;
            
            // Tensor object constructor.
            Tensor();
        };

        // ===================================================================================================
        // Array object.
        //
        // vector<vector<float>> array = 2d vector input data.
        // int width = width of array.
        // int height = height of array.
        // ===================================================================================================
        class Array
        {
            public:
                vector<vector<float>> array;
                Tensor tensor;

                // Array object constructors.
                Array();
                Array(vector<vector<float>> arr);

                // Array object functions.
                Tensor setTensor(vector<vector<float>> arr);
        };

        // ===================================================================================================
        // Derived array object, creates tensor of (w, h) dimensions filled with zeros.
        // ===================================================================================================
        class Zeros : public Array
        {
            public:
                // Zero array object constructors.
                Zeros();
                Zeros(int w, int h);
                Zeros(Tensor t);
        };

        // ===================================================================================================
        // Derived array object, creates tensor of (w, h) dimensions filled with ones.
        // ===================================================================================================
        class Ones : public Array
        {
            public:
                // Ones array object constructors.
                Ones();
                Ones(int w, int h);
                Ones(Tensor t);
        };
    }
}