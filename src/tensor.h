// ===============================================
// Author: Adam Loeckle
// Date: 10/3/2022
// Description: Tensor object header file.
// ===============================================

#include <vector>
using namespace std;

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
            // Tensor object constructors.
            // ===============================================
            Tensor();
            Tensor(vector<vector<float>> matrix);

            // ===============================================
            // Tensor object functions.
            // ===============================================
            int setWidth();
            int setHeight();
            void getTensor();
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
            Zeros() = default;

            // ===============================================
            // Input (width, height) tensor constructor.
            //
            // int w = Desired width of tensor.
            // int h = Desired height of tensor.
            // ===============================================
            Zeros(int w, int h) : width(w), height(h) {tensor = vector<vector<float>> (width, vector<float> (height, 0.0));}
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