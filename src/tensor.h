// ===============================================
// Author: Adam Loeckle
// Date: 10/3/2022
// Description: Tensor object.
// ===============================================

#include <vector>

namespace Grand
{
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
            // Print helper function, prints tensor data in matrix format.
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

    // ===============================================
    // Derived tensor object, creates tensor of (w, h) dimensions filled with zeros.
    //
    // vector<vector<float>> tensor = 2d vector tensor.
    // int width = width of matrix.
    // int height = height of matrix.
    // ===============================================
    class Zeros : public Tensor
    {
        public:
            // ===============================================
            // Default tensor constructor, nothing.
            // ===============================================
            Ones() = default;

            // ===============================================
            // Input (width, height) tensor constructor.
            //
            // int w = Desired width of tensor.
            // int h = Desired height of tensor.
            // ===============================================
            Ones(int w, int h) : width(w), height(h) {tensor = vector<vector<float>> (width, vector<float> (height, 0.0));}
    };

    // ===============================================
    // Derived tensor object, creates tensor of (w, h) dimensions filled with ones.
    //
    // vector<vector<float>> tensor = 2d vector tensor.
    // int width = width of matrix.
    // int height = height of matrix.
    // ===============================================
    class Ones : public Tensor
    {
        public:
            // ===============================================
            // Default tensor constructor, nothing.
            // ===============================================
            Ones() = default;

            // ===============================================
            // Input (width, height) tensor constructor.
            //
            // int w = Desired width of tensor.
            // int h = Desired height of tensor.
            // ===============================================
            Ones(int w, int h) : width(w), height(h) {tensor = vector<vector<float>> (width, vector<float> (height, 1.0));}
    };
}