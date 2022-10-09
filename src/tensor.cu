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
        // vector<vector<float>> tensor = 2d vector tensor.
        // int width = width of matrix.
        // int height = height of matrix.
        // ===============================================

        // ===============================================
        // Default tensor constructor, nothing.
        // ===============================================
        Matrix::Matrix() = default;

        // ===============================================
        // Input matrix tensor constructor.
        //
        // vector<vector<float>> matrix = 2d matrix input.
        // tensor(matrix) = Initializes the value of vector<vector<float>> tensor to matrix input.
        // ===============================================
        Matrix::Matrix(vector<vector<float>> matrix) : matrix(matrix)
        {
            width = setWidth();
            height = setHeight();
            tensor = setTensor(matrix);
        }

        // ===============================================
        // Set helper function, sets tensor object width.
        // ===============================================
        int Matrix::setWidth()
        {
            return matrix.size();
        }

        // ===============================================
        // Set helper function, sets tensor object height.
        // ===============================================
        int Matrix::setHeight()
        {
            return matrix[0].size();
        }

        // ===============================================
        // 2d Vector -> 2d Array converter function for CUDA kernel methods.
        // ===============================================
        float* Matrix::setTensor(vector<vector<float>> mat)
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
        // Derived tensor object, creates tensor of (w, h) dimensions filled with zeros.
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
            matrix = vector<vector<float>> (width, vector<float> (height, 0.0));
            tensor = setTensor(matrix);
        }

        // ===============================================
        // Derived tensor object, creates tensor of (w, h) dimensions filled with ones.
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
            matrix = vector<vector<float>> (width, vector<float> (height, 1.0));
            tensor = setTensor(matrix);
        }
    }
}