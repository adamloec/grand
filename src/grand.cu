#include "grand.h"

// ===================================================================================================
// Main driver test function.
//
// TO RUN:
// nvcc math.cu tensor.cu -o math
// compute-sanitizer .\math.exe (For debugging)
// ===================================================================================================
using namespace Grand;
int main()
{
    vector<vector<float>> data{{1, 2}, {3, 4}, {5, 6}};
    Tensor::Array a(data);
    Tensor::Array b(data);
    Tensor::Tensor c;

    // Add vectors in parallel.
    c = add(a.tensor, b.tensor, 0);
    if (c.status != 1)
    {
        fprintf(stderr, "ERROR: Addition failed.\n");
        return 1;
    }

    // Output
    for (int i = 0; i < c.tensor.width*c.tensor.height; i++)
    {
        cout << "C: " << c.tensor.data[i];
        cout << endl;
    }

    return 0;
}