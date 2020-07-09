// 《并行计算——结构·算法·编程》 P425 15.2.6
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>

using namespace std;

int main(int argc, char **argv)
{
    int rank, size;
    const int n = atoi(argv[1]); // 获取级数规模

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double startTime = MPI_Wtime();

    double local = 0, pi;
    for (int i = rank; i < n; i = i + size)
    {
        double x = (i + 0.5) / n;
        local += 4.0 / (1.0 + pow(x, 2));
    }
    // cout << "progress " << rank << "'s local = " << local << endl;

    MPI_Reduce(&local, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // NOTE:
    // 归约 Reduce 操作对每个进程的发送缓冲区 &local 中的数据按给定的操作进行运算，
    // 并将最终结果存放在 0 进程的接受缓冲区 &pi 中。
    // 参与计算操作的数据项的数据类型为 MPI_DOUBLE ，数据（向量）长度为 1 ，
    // 归约操作为 MPI_SUM 即求和

    pi /= n;

    double endTime = MPI_Wtime();

    if (rank == 0)
    {
        cout << "pi = " << pi << endl;
        cout << "Elapsed time = " << endTime - startTime << "s" << endl;
    }

    MPI_Finalize();
    return 0;
}