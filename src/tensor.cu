// ===================================================================================================
// Author: Adam Loeckle
// Date: 10/3/2022
// Description: Tensor object source file.
// ===================================================================================================

#ifndef TENSOR_INCL
#define TENSOR_INCL
    #include "tensor.h"
#endif

namespace Grand
{
    namespace Tensor
    {
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

        // Default tensor constructor, nothing.
        Tensor::Tensor() = default;

        // ===================================================================================================
        // Array object.
        //
        // vector<vector<float>> matrix = 2d vector.
        // int width = width of matrix.
        // int height = height of matrix.
        // float *tensor = tensor used for kernel functions.
        // ===================================================================================================

        // Default array constructor, nothing.
        Array::Array() = default;

        // Input matrix tensor constructor.
        //
        // vector<vector<float>> matrix = 2d matrix input.
        // tensor(matrix) = Initializes the value of vector<vector<float>> tensor to matrix input.
        //
        // Vector parameter specifically for gathering matrix dimensions for kernel use.
        Array::Array(vector<vector<float>> array) : array(array)
        {
            tensor = setTensor(array);
        }

        // 2d Vector -> Tensor converter function for CUDA kernel methods.
        Tensor Array::setTensor(vector<vector<float>> mat)
        {
            Tensor t;
            t.width = array.size();
            t.height = array[0].size();

            t.data = new float[t.width*t.height];
            int count = 0;
            for (int i = 0; i < t.width; i++)
            {
                for (int j = 0; j < t.height; j++)
                {
                    t.data[count] = mat[i][j];
                    count++;
                }
            }

            return t;
        }

        // ===================================================================================================
        // Zeros array object, creates tensor object of (w, h) dimensions filled with zeros.
        // ===================================================================================================
        
        // Default zeros array constructor, nothing.
        Zeros::Zeros() = default;

        // Input (width, height) zeros array constructor.
        //
        // int w = Desired width of tensor.
        // int h = Desired height of tensor.
        Zeros::Zeros(int w, int h) 
        {
            array = vector<vector<float>> (w, vector<float> (h, 0.0));
            tensor = setTensor(array);
        }
        
        // Input (tensor) zeros array constructor.
        //
        // Tensor t = tensor input, copies tensor properties to generate zeros tensor of size (t.width. t.height).
        Zeros::Zeros(Tensor t)
        {
            array = vector<vector<float>> (t.width, vector<float> (t.height, 0.0));
            tensor = setTensor(array);
        }

        // ===================================================================================================
        // Ones array object, creates tensor of (w, h) dimensions filled with ones.
        // ===================================================================================================

        // Default ones array constructor, nothing.
        Ones::Ones() = default;

        // Input (width, height) ones array constructor.
        //
        // int w = Desired width of tensor.
        // int h = Desired height of tensor.
        Ones::Ones(int w, int h)
        {
            array = vector<vector<float>> (w, vector<float> (h, 1.0));
            tensor = setTensor(array);
        }

        // Input (tensor) ones array constructor.
        //
        // Tensor t = tensor input, copies tensor properties to generate ones tensor of size (t.width. t.height).
        Ones::Ones(Tensor t)
        {
            array = vector<vector<float>> (t.width, vector<float> (t.height, 1.0));
            tensor = setTensor(array);
        }
    }
}