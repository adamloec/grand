#include "grand.h"

int main()
{
    vector<vector<float>> data{{1, 2}, {3, 4}};
    Tensor a(data);
    Tensor b(1, 4, 4);
    a.print();
    b.print();
    return 0;
}