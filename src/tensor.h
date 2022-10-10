// ===================================================================================================
// Author: Adam Loeckle
// Date: 10/3/2022
// Description: Tensor object header file.
// ===================================================================================================

#ifndef CORE_INCL
#define CORE_INCL
    #include "cuda_runtime.h"
    #include "device_launch_parameters.h"
    #include <stdio.h>
    #include <iostream>
    #include <vector>
    using namespace std;
#endif

namespace Grand
{
    namespace Tensor
    {
        // Matrix theory
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
            
            // Default tensor constructor, nothing.
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

                // Default array constructor, nothing.
                Array();

                // Input matrix tensor constructor.
                //
                // vector<vector<float>> matrix = 2d matrix input.
                // tensor(matrix) = Initializes the value of vector<vector<float>> tensor to matrix input.
                //
                // Vector parameter specifically for gathering matrix dimensions for kernel use.
                Array(vector<vector<float>> arr);

                // Function to create Tensor object out of Array data for kernel use.
                //
                // vector<vector<float>> arr = Array.array data
                //
                // Return: Tensor
                Tensor setTensor(vector<vector<float>> arr);
        };

        // ===================================================================================================
        // Derived array object, creates tensor of (w, h) dimensions filled with zeros.
        // ===================================================================================================
        class Zeros : public Array
        {
            public:
                // Default zeros array constructor, nothing.
                Zeros();

                // Input (width, height) zeros array constructor.
                //
                // int w = Desired width of tensor.
                // int h = Desired height of tensor.
                Zeros(int w, int h);

                // Input (tensor) zeros array constructor.
                //
                // Tensor t = tensor input, copies tensor properties to generate zeros tensor of size (t.width. t.height).
                Zeros(Tensor t);
        };

        // ===================================================================================================
        // Derived array object, creates tensor of (w, h) dimensions filled with ones.
        // ===================================================================================================
        class Ones : public Array
        {
            public:
                // Default ones array constructor, nothing.
                Ones();

                // Input (width, height) ones array constructor.
                //
                // int w = Desired width of tensor.
                // int h = Desired height of tensor.
                Ones(int w, int h);

                // Input (tensor) ones array constructor.
                //
                // Tensor t = tensor input, copies tensor properties to generate ones tensor of size (t.width. t.height).
                Ones(Tensor t);
        };
    }
}