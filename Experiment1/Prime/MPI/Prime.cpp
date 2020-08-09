#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv)
{
    int rank, size;
    const int n = atoi(argv[1]); // 获取级数规模

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double startTime = MPI_Wtime();

    bool isComposite[n + 1] = {true, true}; // 0 1 是合数

    for (int i = 2; i <= (int)sqrt(n); i++)
        if (!isComposite[i])
            for (int j = i; i * j <= n; j++)
                isComposite[i * j] = true;

    int local = 0, total;

    for (int i = 2 * rank + 1; i <= n; i += 2 * size)
        if (!isComposite[i])
            local++;

    MPI_Reduce(&local, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // NOTE:
    // 归约 Reduce 操作对每个进程的发送缓冲区 &local 中的数据按给定的操作进行运算，
    // 并将最终结果存放在 0 进程的接受缓冲区 &total 中。
    // 参与计算操作的数据项的数据类型为 MPI_INT ，数据（向量）长度为 1 ，
    // 归约操作为 MPI_SUM 即求和

    if (n >= 2)
        total++;

    double endTime = MPI_Wtime();

    if (rank == 0)
    {
        cout << "The number of prime numbers is " << total << endl;
        cout << "Elapsed time = " << 1E6 * (endTime - startTime) << "microsecond" << endl;
    }

    MPI_Finalize();
    return 0;
}