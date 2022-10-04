#include "grand.h"

int main()
{
    float **data = new float*[2];
    for (int i = 0; i < 2; i++)
    {
        data[i] = new float[2];
    }
    data[0][0] = 1.0;
    data[0][1] = 2.0;
    data[1][0] = 3.0;
    data[1][1] = 4.0;

    Tensor a(data);
    a.print();
    return 0;
}