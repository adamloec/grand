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

        Tensor::Tensor() = default;

        // ===================================================================================================
        // Array object.
        //
        // vector<vector<float>> matrix = 2d vector.
        // int width = width of matrix.
        // int height = height of matrix.
        // float *tensor = tensor used for kernel functions.
        // ===================================================================================================

        Array::Array() = default;

        Array::Array(vector<vector<float>> array) : array(array)
        {
            tensor = setTensor(array);
        }

        Tensor Array::setTensor(vector<vector<float>> mat)
        {
            // Create tensor object, set width and height.
            Tensor t;
            t.width = array.size();
            t.height = array[0].size();

            // Iterate through 2d vector, fill tensor 1d array with values from vector.
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
        
        Zeros::Zeros() = default;

        Zeros::Zeros(int w, int h) 
        {
            array = vector<vector<float>> (w, vector<float> (h, 0.0));
            tensor = setTensor(array);
        }
        
        Zeros::Zeros(Tensor t)
        {
            array = vector<vector<float>> (t.width, vector<float> (t.height, 0.0));
            tensor = setTensor(array);
        }

        // ===================================================================================================
        // Ones array object, creates tensor of (w, h) dimensions filled with ones.
        // ===================================================================================================

        Ones::Ones() = default;

        Ones::Ones(int w, int h)
        {
            array = vector<vector<float>> (w, vector<float> (h, 1.0));
            tensor = setTensor(array);
        }

        Ones::Ones(Tensor t)
        {
            array = vector<vector<float>> (t.width, vector<float> (t.height, 1.0));
            tensor = setTensor(array);
        }
    }
}