// ===============================================
// Author: Adam Loeckle
// Date: 10/3/2022
// Description: Tensor object source file.
// ===============================================

#include "tensor.h"

namespace Grand
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
    Tensor::Tensor() = default;

    // ===============================================
    // Input matrix tensor constructor.
    //
    // vector<vector<float>> matrix = 2d matrix input.
    // tensor(matrix) = Initializes the value of vector<vector<float>> tensor to matrix input.
    // ===============================================
    Tensor::Tensor(vector<vector<float>> matrix) : tensor(matrix)
    {
        width = setWidth();
        height = setHeight();
    }

    // ===============================================
    // Set helper function, sets tensor object width.
    // ===============================================
    int Tensor::setWidth()
    {
        return tensor.size();
    }

    // ===============================================
    // Set helper function, sets tensor object height.
    // ===============================================
    int Tensor::setHeight()
    {
        return tensor[0].size();
    }

    // ===============================================
    // Print helper function, prints tensor data in matrix format.
    // ===============================================
    void Tensor::getTensor()
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
}