#include "grand.h"

// ===================================================================================================
// Main driver test function.
//
// TO RUN:
// nvcc grand.cu math.cu tensor.cu -o math
// compute-sanitizer .\math.exe (For debugging)
// ===================================================================================================
using namespace Grand;
int main()
{
    vector<vector<float>> data1{{1, 2, 3}, {4, 5, 6}};
    vector<vector<float>> data2{{1, 2}, {3, 4}, {5, 6}};
    Tensor::Array a(data1);
    Tensor::Array b(data2);

    // Add vectors in parallel.
    Tensor::Tensor c = dot(a.tensor, b.tensor, 0);
    if (c.status != 1)
    {
        fprintf(stderr, "ERROR: Addition failed.\n");
        return 1;
    }

    // Output
    for (int i = 0; i < c.width*c.height; i++)
    {
        cout << "C: " << c.data[i];
        cout << endl;
    }

    return 0;
}