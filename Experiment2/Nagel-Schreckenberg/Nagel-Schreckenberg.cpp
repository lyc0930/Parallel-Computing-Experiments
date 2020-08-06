// 《并行计算——结构·算法·编程》 P425 15.2.6
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <mpi.h>

using namespace std;

const double p = 0.2;         // probability of slow down
const unsigned int v_max = 8; // upper bound of velocity

int main(int argc, char *argv[])
{
    int rank, size;
    const int n = atoi(argv[1]);              // 获取车辆数量
    const int numberOfCycles = atoi(argv[2]); // 周期数

    unsigned int velocity[n + 1];
    unsigned int position[n + 1];

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand((unsigned)(time(NULL) + rank)); // 随机种子

    double startTime = MPI_Wtime();

    int *recvcounts =
        (int *)malloc(size * sizeof(int)); // integer array (of length group size) containing the number of elements
                                           // that are received from each process (significant only at root)
    int *displs = (int *)malloc(
        size * sizeof(int)); // integer array (of length group size). Entry i specifies the displacement relative to
                             // recvbuf at which to place the incoming data from process i (significant only at root)

    for (int i = 0; i < size; i++)
    {
        displs[i] = i * n / size;                                               // 局部队首
        recvcounts[i] = ((i == size - 1) ? n : (i + 1) * n / size) - displs[i]; // 局部长度
    }

    int head = displs[rank];                    // 局部队首
    int tail = displs[rank] + recvcounts[rank]; // 局部队尾

    for (int i = head; i < tail; i++)
    {
        velocity[i] = 0;
        position[i] = i + 1;
    }
    position[n] = numeric_limits<unsigned int>::max();

    for (int i = 0; i < numberOfCycles; i++)
    {
        if (rank != 0) // 不为最后一部分
            MPI_Send(position + head, 1, MPI_UNSIGNED, rank - 1, i * (size - 1) + (rank - 1), MPI_COMM_WORLD);
        if (rank != size - 1) // 不为首个部分
        {
            MPI_Status stat;
            MPI_Recv(position + tail, 1, MPI_UNSIGNED, rank + 1, i * (size - 1) + rank, MPI_COMM_WORLD, &stat);
        }
        for (int j = head; j < tail; j++)
        {
            unsigned int distance = position[j + 1] - position[j] - 1; // 车辆间距

            // Shift
            if (velocity[j] < distance)
            {
                if (velocity[j] < v_max)
                    velocity[j]++;
            }
            else
                velocity[j] = distance;

            if (rand() / (double)RAND_MAX < p) // 随机减速
                if (velocity[j] > 0)
                    velocity[j]--;

            position[j] += velocity[j];
        }
    }

    unsigned int velocity_final[n + 1];
    unsigned int position_final[n + 1];

    MPI_Gatherv(velocity + head, tail - head, MPI_UNSIGNED, velocity_final, recvcounts, displs, MPI_UNSIGNED, 0,
                MPI_COMM_WORLD);
    MPI_Gatherv(position + head, tail - head, MPI_UNSIGNED, position_final, recvcounts, displs, MPI_UNSIGNED, 0,
                MPI_COMM_WORLD);

    double endTime = MPI_Wtime();

    free(recvcounts);
    free(displs);

    if (rank == 0)
    {
        ofstream outFile;
        outFile.open("Result.data", ios::out | ios::trunc);
        outFile << "n = " << n << endl;
        outFile << "Cycles = " << numberOfCycles << endl;
        for (int i = n - 1; i >= 0; i--)
            outFile << "Car " << n - i - 1 << " : Position " << position_final[i] << ", Velocity " << velocity_final[i]
                    << endl;
        outFile.close();
        cout << "Elapsed time = " << endTime - startTime << "s" << endl;
    }

    MPI_Finalize();
    return 0;
}