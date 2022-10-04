#include "grand.h"

int main()
{
    vector<vector<float>> data{{1, 2}, {3, 4}};
    Grand::Tensor a(data);
    Grand::Zeros b(5, 5);
    a.print();
    b.print();
    return 0;
}