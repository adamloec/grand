// ===============================================
// Author: Adam Loeckle
// Date: 10/3/2022
// Description: Tensor object source file.
// ===============================================

#ifndef TENSOR_INCL
#define TENSOR_INCL
    #include "tensor.h"
#endif

namespace Grand
{
    namespace Tensor
    {
        // ===============================================
        // Tensor object.
        //
        // vector<vector<float>> matrix = 2d vector.
        // int width = width of matrix.
        // int height = height of matrix.
        // float *tensor = tensor used for kernel functions.
        // ===============================================

        // ===============================================
        // Default tensor constructor, nothing.
        // ===============================================
        Array::Array() = default;

        // ===============================================
        // Input matrix tensor constructor.
        //
        // vector<vector<float>> matrix = 2d matrix input.
        // tensor(matrix) = Initializes the value of vector<vector<float>> tensor to matrix input.
        //
        // Vector parameter specifically for gathering matrix dimensions for kernel use.
        // ===============================================
        Array::Array(vector<vector<float>> array) : array(array)
        {
            width = setWidth();
            height = setHeight();
            tensor = setTensor(array);
        }

        // ===============================================
        // Set helper function, sets tensor object width.
        // ===============================================
        int Array::setWidth()
        {
            return array.size();
        }

        // ===============================================
        // Set helper function, sets tensor object height.
        // ===============================================
        int Array::setHeight()
        {
            return array[0].size();
        }

        // ===============================================
        // 2d Vector -> 2d Array converter function for CUDA kernel methods.
        // ===============================================
        float* Array::setTensor(vector<vector<float>> mat)
        {
            float *temp = new float[width*height];
            int count = 0;
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    temp[count] = mat[i][j];
                    count++;
                }
            }
            return temp;
        }

        // ===============================================
        // Zeros tensor object, creates tensor of (w, h) dimensions filled with zeros.
        // ===============================================
        
        // ===============================================
        // Default tensor constructor, nothing.
        // ===============================================
        Zeros::Zeros() = default;

        // ===============================================
        // Input (width, height) tensor constructor.
        //
        // int w = Desired width of tensor.
        // int h = Desired height of tensor.
        // ===============================================
        Zeros::Zeros(int w, int h) 
        {
            width = w; 
            height = h;
            array = vector<vector<float>> (width, vector<float> (height, 0.0));
            tensor = setTensor(array);
        }

        // ===============================================
        // Ones tensor object, creates tensor of (w, h) dimensions filled with ones.
        // ===============================================

        // ===============================================
        // Default tensor constructor, nothing.
        // ===============================================
        Ones::Ones() = default;

        // ===============================================
        // Input (width, height) tensor constructor.
        //
        // int w = Desired width of tensor.
        // int h = Desired height of tensor.
        // ===============================================
        Ones::Ones(int w, int h)
        {
            width = w; 
            height = h; 
            array = vector<vector<float>> (width, vector<float> (height, 1.0));
            tensor = setTensor(array);
        }
    }
}