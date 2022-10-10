#include "grand.h"

using namespace Grand;
int main()
{
    vector<vector<float>> data{{1, 2, 3}, {3, 4, 5}, {5, 6, 7}};
    Tensor::Array a(data);
    return 0;
}