// Tensor struct
// width: width of matrix
// height: height of matrix
// data: 2d matrix
class Tensor
{
    public:
        float **data = NULL;
        int width = 0;
        int height = 0;

        Tensor() = default;
        Tensor(float **matrix) : data(matrix) {width = getWidth(); height = getHeight();}

        int getWidth(void)
        {
            return sizeof(data)/sizeof(data[0]);
        }

        int getHeight(void)
        {
            return sizeof(data[0])/sizeof(float);
        }

        void print()
        {
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    cout << data[i][j] << " ";
                }
                cout << endl;
            }
        }
};