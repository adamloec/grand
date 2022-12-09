#include "grand.h"

// ===================================================================================================
// Main driver test function.
//
// TO RUN:
// nvcc grand.cu math.cu tensor.cu -o grand
// .\grand.exe
//
// compute-sanitizer .\grand.exe (For debugging)
// ===================================================================================================
namespace Grand
{
    
}
using namespace Grand;
int main()
{
    vector<vector<float>> data1{{1, 2, 3, 4}, {5, 6, 7, 8}};
    vector<vector<float>> data2{{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    Tensor::Array a(data1);
    Tensor::Array b(data2);

    // Multiply vectors.
    Tensor::Tensor c = dot(a.tensor, b.tensor, 0);

    // Output
    for (int i = 0; i < c.width*c.height; i++)
    {
        cout << "C: " << c.data[i];
        cout << endl;
    }

    return 0;
}