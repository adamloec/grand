#include "grand.h"

int main()
{
    vector<vector<float>> data{{1, 2, 3}, {3, 4, 5}, {5, 6, 7}};
    Grand::Tensor::Matrix a(data);
    return 0;
}