#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <mpi.h>

using namespace std;

const double p = 0.2;          // probability of slow down
const unsigned int v_max = 12; // upper bound of velocity

int main(int argc, char* argv[])
{
    int rank, size;
    const int n = atoi(argv[1]);              // 获取车辆数量
    const int numberOfCycles = atoi(argv[2]); // 周期数

    unsigned int velocity[n];     // 速度
    unsigned int position[n + 1]; // 位置

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand((unsigned)(time(NULL) + rank)); // 随机种子

    double startTime = MPI_Wtime();

    int recvcounts[size]; // integer array (of length group size) containing the number of elements
                          // that are received from each process (significant only at root)
    int displs[size];     // integer array (of length group size). Entry i specifies the displacement relative to
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

    position[n] = numeric_limits<unsigned int>::max(); // 最后一辆车前方没有车辆

    for (int i = 0; i < numberOfCycles; i++)
    {
        if (rank != 0) // 不为最后一部分
            MPI_Send(position + head, 1, MPI_UNSIGNED, rank - 1, i * (size - 1) + (rank - 1), MPI_COMM_WORLD);
        if (rank != size - 1) // 不为首个部分
        {
            MPI_Status status;
            MPI_Recv(position + tail, 1, MPI_UNSIGNED, rank + 1, i * (size - 1) + rank, MPI_COMM_WORLD, &status);
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

    if (rank == 0)
    {
        MPI_Gatherv(MPI_IN_PLACE, tail - head, MPI_UNSIGNED, velocity, recvcounts, displs, MPI_UNSIGNED, 0,
                    MPI_COMM_WORLD);
        MPI_Gatherv(MPI_IN_PLACE, tail - head, MPI_UNSIGNED, position, recvcounts, displs, MPI_UNSIGNED, 0,
                    MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gatherv(velocity + head, tail - head, MPI_UNSIGNED, velocity, recvcounts, displs, MPI_UNSIGNED, 0,
                    MPI_COMM_WORLD);
        MPI_Gatherv(position + head, tail - head, MPI_UNSIGNED, position, recvcounts, displs, MPI_UNSIGNED, 0,
                    MPI_COMM_WORLD);
    }

    double endTime = MPI_Wtime();

    if (rank == 0)
    {
        ofstream outFile;
        outFile.open(string("../traffic_condition/") + argv[1] + "x" + argv[2] + ".data", ios::out | ios::trunc);
        outFile << "n = " << n << endl;
        outFile << "Cycles = " << numberOfCycles << endl;
        for (int i = n - 1; i >= 0; i--)
            outFile << "Car " << n - i - 1 << " : Position " << position[i] << ", Velocity " << velocity[i] << endl;
        outFile.close();
        cout << "Elapsed time = " << endTime - startTime << "s" << endl;
    }

    MPI_Finalize();
    return 0;
}