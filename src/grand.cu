#include "grand.h"

int main()
{
    vector<vector<float>> data{{1, 2}, {3, 4}};
    Grand::Tensor::Matrix a(data);
    a.getTensor();
    return 0;
}