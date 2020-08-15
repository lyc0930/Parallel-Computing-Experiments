#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <mpi.h>
#define N 256

using namespace std;

// 单位统一
long double operator"" _cm(long double l) { return l / 100; }
long double operator"" _cm(unsigned long long l) { return l / 100.0; }

long double operator"" _m(long double l) { return l; }

long double operator"" _kg(long double m) { return m; }
long double operator"" _kg(unsigned long long m) { return m; }

long double operator"" _s(long double t) { return t; }
long double operator"" _s(unsigned long long t) { return t; }

long double operator"" _ms(long double t) { return t / 1000; }
long double operator"" _ms(unsigned long long t) { return t / 1000.0; }

const long double G = 6.6710E-11; // 引力常数
const long double m = 10000_kg;   // 小球质量
const long double deltaT = 1_ms;  // 刷新率
const long double T = 10_s;       // 总仿真时间

long double Coordinate_x[N], Coordinate_y[N]; // 坐标
long double Velocity_x[N], Velocity_y[N];     // 速度
long double Force_x[N], Force_y[N];           // 受力

// 在每个并行进程中计算小球数据
class ParallelBodiesCalculator
{
  private:
    int n;
    int head, tail;

  public:
    ParallelBodiesCalculator(int n, int head, int tail) : n(n), head(head), tail(tail){};

    // 计算每个小球受到的作用力
    void force()
    {
        for (int i = head; i < tail; i++)
            Force_x[i] = Force_y[i] = 0;

        for (int i = head; i < tail; i++)
        {
            for (int j = 0; j < n; j++) // 计算其他小球作用力在坐标轴上的分量
                if (i != j)
                {
                    long double dx = Coordinate_x[j] - Coordinate_x[i];
                    long double dy = Coordinate_y[j] - Coordinate_y[i];
                    long double r = sqrtl(dx * dx + dy * dy); // 两球间距离

                    if (r == 0.0)
                        r = numeric_limits<long double>::min(); // 防止除以 0

                    Force_x[i] += G * powl(m, 2) * dx / powl(r, 3);
                    Force_y[i] += G * powl(m, 2) * dy / powl(r, 3);
                }
        }
        return;
    }

    // 计算每个小球的速度
    void velocities()
    {
        force();
        for (int i = head; i < tail; i++) // 计算每个小球的速度改变量
        {
            Velocity_x[i] += Force_x[i] / m * deltaT;
            Velocity_y[i] += Force_y[i] / m * deltaT;
        }
        return;
    }

    // 计算每个小球的位置
    void positions()
    {
        velocities();
        for (int i = head; i < tail; i++) // 计算每个小球的位置改变量
        {
            Coordinate_x[i] += Velocity_x[i] * deltaT;
            Coordinate_y[i] += Velocity_y[i] * deltaT;
        }
        return;
    }

    // 计算每个小球的数据
    void calculate()
    {
        positions();
        return;
    }
};

int main(int argc, char* argv[])
{
    int rank, size;
    const int n = atoi(argv[1]); // 获取小球数量

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double startTime = MPI_Wtime();

    for (int i = 0; i < n; i++)
    {
        Velocity_x[i] = Velocity_y[i] = 0;
        Coordinate_x[i] = (i % (int)sqrt(n)) * 1_cm;
        Coordinate_y[i] = (i / (int)sqrt(n)) * 1_cm;
    }

    int recvcounts[size];
    int displs[size];

    for (int i = 0; i < size; i++)
    {
        displs[i] = i * n / size;                                               // 局部队首
        recvcounts[i] = ((i == size - 1) ? n : (i + 1) * n / size) - displs[i]; // 局部长度
    }

    int head = displs[rank];                    // 局部队首
    int tail = displs[rank] + recvcounts[rank]; // 局部队尾

    ParallelBodiesCalculator calculator(n, head, tail);

    for (long double t = 0; t < T; t += deltaT)
    {
        calculator.calculate();

        MPI_Allgatherv(MPI_IN_PLACE, tail - head, MPI_LONG_DOUBLE, Coordinate_x, recvcounts, displs, MPI_LONG_DOUBLE,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, tail - head, MPI_LONG_DOUBLE, Coordinate_y, recvcounts, displs, MPI_LONG_DOUBLE,
                       MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        MPI_Gatherv(MPI_IN_PLACE, tail - head, MPI_LONG_DOUBLE, Velocity_x, recvcounts, displs, MPI_LONG_DOUBLE, 0,
                    MPI_COMM_WORLD);
        MPI_Gatherv(MPI_IN_PLACE, tail - head, MPI_LONG_DOUBLE, Velocity_y, recvcounts, displs, MPI_LONG_DOUBLE, 0,
                    MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gatherv(Velocity_x + head, tail - head, MPI_LONG_DOUBLE, Velocity_x, recvcounts, displs, MPI_LONG_DOUBLE, 0,
                    MPI_COMM_WORLD);
        MPI_Gatherv(Velocity_y + head, tail - head, MPI_LONG_DOUBLE, Velocity_y, recvcounts, displs, MPI_LONG_DOUBLE, 0,
                    MPI_COMM_WORLD);
    }

    double endTime = MPI_Wtime();

    if (rank == 0)
    {
        ofstream outFile;
        outFile.open(string("../n-body_result/") + argv[1] + "-body.data", ios::out | ios::trunc);
        outFile << n << " bodies after simulation of " << T << " seconds (1 frame per " << deltaT << " microsecond)"
                << endl;

        for (int i = 0; i < n; i++)
            outFile << "Body " << i + 1 << " : Position (" << Coordinate_x[i] << "m, " << Coordinate_y[i]
                    << "m), Velocity (" << Velocity_x[i] << "m/s, " << Velocity_y[i] << "m/s)" << endl;

        outFile.close();

        cout << "Elapsed time = " << endTime - startTime << "s" << endl;
    }

    MPI_Finalize();
    return 0;
}