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
        Matrix::Matrix(vector<vector<float>> matrix) : tensor(matrix)
        {
            width = setWidth();
            height = setHeight();
        }

        // ===============================================
        // Set helper function, sets tensor object width.
        // ===============================================
        int Matrix::setWidth()
        {
            return tensor.size();
        }

        // ===============================================
        // Set helper function, sets tensor object height.
        // ===============================================
        int Matrix::setHeight()
        {
            return tensor[0].size();
        }

        // ===============================================
        // Print helper function, prints tensor data in matrix format.
        // ===============================================
        void Matrix::getTensor()
        {
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    cout << tensor[i][j] << " ";
                }
                cout << endl;
            }
        }

        // ===============================================
        // Derived tensor object, creates tensor of (w, h) dimensions filled with zeros.
        //
        // vector<vector<float>> tensor = 2d vector tensor.
        // int width = width of matrix.
        // int height = height of matrix.
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
            tensor = vector<vector<float>> (width, vector<float> (height, 0.0));
        }

        // ===============================================
        // Derived tensor object, creates tensor of (w, h) dimensions filled with ones.
        //
        // vector<vector<float>> tensor = 2d vector tensor.
        // int width = width of matrix.
        // int height = height of matrix.
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
            tensor = vector<vector<float>> (width, vector<float> (height, 1.0));
        }
    }
}