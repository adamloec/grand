// ===============================================
// Author: Adam Loeckle
// Date: 10/3/2022
// Description: Tensor object.
// ===============================================

#include <vector>

// ===============================================
// Tensor object.
//
// vector<vector<float>> tensor = 2d vector tensor.
// int width = width of matrix.
// int height = height of matrix.
// ===============================================
class Tensor
{
    public:
        vector<vector<float>> tensor;
        int width = 0;
        int height = 0;

        // ===============================================
        // Default tensor constructor, nothing.
        // ===============================================
        Tensor() = default;

        // ===============================================
        // Input matrix tensor constructor.
        //
        // vector<vector<float>> matrix = 2d matrix input.
        // tensor(matrix) = Initializes the value of vector<vector<float>> tensor to matrix input.
        // ===============================================
        Tensor(vector<vector<float>> matrix) : tensor(matrix) {width = setWidth(); height = setHeight();}
        
        // ===============================================
        // Zeros and ones tensor constructor.
        //
        // int c = Variable input, 0 = zeros tensor, 1 = ones tensor.
        // int w = Desired width of tensor.
        // int h = Desired height of tensor.
        // ===============================================
        Tensor(int c, int w, int h) : width(w), height(h)
        {
            if (c == 0) { tensor = vector<vector<float>> (width, vector<float> (height, 0.0)); }

            if (c == 1) { tensor = vector<vector<float>> (width, vector<float> (height, 1.0)); }
        }

        // ===============================================
        // Set helper function, sets tensor object width.
        // ===============================================
        int setWidth(void)
        {
            return tensor.size();
        }

        // ===============================================
        // Set helper function, sets tensor object height.
        // ===============================================
        int setHeight(void)
        {
            return tensor[0].size();
        }

        // ===============================================
        // Print tensor data in matrix format.
        // ===============================================
        void print()
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
};