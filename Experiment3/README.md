# <center>实验 3</center>

##### <p align="right">PB17000297 罗晏宸</br>2020.8.7</p>

### 实验题目

利用 MPI 解决 N 体问题

N体问题是指找出已知初始位置、速度和质量的多个物体在经典力学情况下的后续运动。在本次实验中，你需要模拟N个物体在二维空间中的运动情况。通过计算每两个物体之间的相互作用力，可以确定下一个时间周期内的物体位置。在本次实验中，初始情况下，$N$ 个小球等间隔分布在一个正方形的二维空间中，小球在运动时没有范围限制。每个小球间会且只会受到其他小球的引力作用。小球可以看成质点。小球移动不会受到其他小球的影响（即不会发生碰撞，挡住等情况）。你需要计算模拟一定时间后小球的分布情况，并通过 MPI 并行化计算过程。

有关参数要求如下：
1. 引力常数数值取 $6.67 \times 10^{-11}$
2. 小球重量都为 $10000 \text{kg}$
3. 小球间的初始间隔为 $1 \text{cm}$ ，例： $N = 36$ 时，初始的正方形区域为 $ 5 \text{cm} \times 5 \text{cm}$
4. 小球初速为 $0$

### 实验环境

#### 操作系统

Windows Subsystem for Linux: Ubuntu 18.04 LTS （内核版本 4.4.0-18362-Microsoft）

#### 编译环境

- g++ 7.5.0

- OpenMP 4.5.0

- MPICH 3.3a2

#### 硬件配置

​	Intel&reg;  Core&trade;  i5-7200U CPU

- 内核数：2
- 线程数：4（超线程）

### 算法设计与分析

N 体问题有如下的朴素算法：在每个时刻对每个小球

1. 由**所有其他小球的位置**计算受力，进而得到加速度
2. 由加速度与速度更新速度
3. 由速度与位置更新位置

而对于并行算法而言，均匀分配每个进程计算不同的小球即可完成上述算法的并行化。注意到计算每个小球的受力需要其他所有小球的位置，因此在每个时刻计算结束之后需要全局收集(Allgatherv)所有进程的位置数据以进行同步。

一个需要明确的算法细节是，对 N 题问题中小球运动进行的模拟是具有时间粒度的，即相邻时刻间具有一定的时间间隔 $\Delta T$，为了模拟的效果我们当然希望 $\Delta T$ 越小越好。在本实验中，我们考虑总仿真时间为 $T = 10 \text{s}$，时间间隔 $\Delta T = 1 \text{ms}$，这相对平衡了程序运行时间与模拟细腻程度。

另一个需要注意的地方是，在本问题中我们将小球看做质点，不考虑任何形式的小球间接触与碰撞，那对于两个小球彼此距离接近为 0 的情况，我们进一步约定：

1. 任意小球间的距离不会降低到 0 ，距离有下限 $r_{\min}$
2. 对于可能发生碰撞的两个小球，它们会忽略体积与位置而彼此穿过（仍考虑相互的作用力）

### 核心代码

#### 变量与字面量设计

由于本问题有实际的问题背景，因此在程序中设置各物理单位的字面量，既避免了可能出现的单位换算错误，也提高了含常量代码的可读性

```c++
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
```

问题发生在二维平面上，考虑到 MPI 通信，对每个小球的坐标、速度、受力在 $x$ 轴与 $y$ 轴上的分量分别设置变量数组

```c++
long double Coordinate_x[N], Coordinate_y[N]; // 坐标
long double Velocity_x[N], Velocity_y[N];     // 速度
long double Force_x[N], Force_y[N];           // 受力
```

#### 计算函数（实验要求）

按实验要求，典型的程序中应当包含如下依次调用的三个函数：

1. `compute force()` ：计算每个小球受到的作用力
2. `compute velocities()` ：计算每个小球的速度
3. `compute positions()` ：计算每个小球的位置

为了减少观感不佳的参数传递，设计一个`ParallelBodiesCalculator`类，上述三个函数是其方法。

```c++
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
```

在每个进程中例化一个对象来完成计算

```c++
    ParallelBodiesCalculator calculator(n, head, tail);
```

#### 初始化

按问题描述，初始化小球的位置与速度，并为每个进程分配小球。

```c++
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
```

#### 周期计算

以时间为循环变量，每个时刻对小球进行计算，并全局收集同步位置信息，这里采用了`MPI_IN_PLACE`常量作为`MPI_Allgatherv`的发送地址，表示数据同步后存放在源地址。
```c++
    for (long double t = 0; t < T; t += deltaT)
    {
        calculator.calculate();

        MPI_Allgatherv(MPI_IN_PLACE, tail - head, MPI_LONG_DOUBLE, Coordinate_x, recvcounts, displs, MPI_LONG_DOUBLE,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, tail - head, MPI_LONG_DOUBLE, Coordinate_y, recvcounts, displs, MPI_LONG_DOUBLE,
                       MPI_COMM_WORLD);
    }
```
周期结束后，`root` 进程收集全局的速度以便输出
```c++
    if (rank == 0)
    {
        MPI_Gatherv(MPI_IN_PLACE, tail - head, MPI_LONG_DOUBLE, Velocity_x, recvcounts, displs, MPI_LONG_DOUBLE, 0,
                    MPI_COMM_WORLD);
        MPI_Gatherv(MPI_IN_PLACE, tail - head, MPI_LONG_DOUBLE, Velocity_y, recvcounts, displs, MPI_LONG_DOUBLE, 0,
                    MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gatherv(Velocity_x, tail - head, MPI_LONG_DOUBLE, Velocity_x, recvcounts, displs, MPI_LONG_DOUBLE, 0,
                    MPI_COMM_WORLD);
        MPI_Gatherv(Velocity_y, tail - head, MPI_LONG_DOUBLE, Velocity_y, recvcounts, displs, MPI_LONG_DOUBLE, 0,
                    MPI_COMM_WORLD);
    }
```

### 实验结果

- 编译运行
程序采用 MPICH 编译运行，在终端相应目录下执行

```shell
mpicxx -g -Wall -o n-body.o n-body.cpp
mpirun -n 4 ./n-body.o 64
```

即以 4 个并行进程运行程序，数据规模为 $n = 64$

-  道路情况
每次程序运行都会在`n-body_result`文件夹下输出文件名形如`N-body.data`的数据文件，其中`N`表示小球数量，文件中的数据条目以`Body i : Position ( ,  ), Velocity ( ,  )`的形式记录每个小球在结束模拟之后的位置与速度（在 $x$ 和 $y$ 方向上的分量）。

-  运行时间(s)

每个数据点的运行时间统计是 10 次运行的平均时间

| 规模（小球数量）\进程数 | 1           | 2           | 4           |
| :---------- | ----------- | ----------- | ----------- |
| 64      | 0.505444  | 0.348680  | 0.312471 |
| 256      |  10.3277 | 5.88557 | 4.23051 |

- 加速比

| 规模（小球数量）\进程数 | 1           | 2           | 4           |
| :---------- | ----------- | ----------- | ----------- |
| 64        | 1  |1.44959 | 1.61757 |
| 256       | 1 | 1.75475 | 2.44124 |

### 分析与总结

本问题实现的通信开销非常直观，每一个时刻都有 $\Theta (2np)$ 全局收集开销，当 $n$ 相对与进程数 $p$ 较小时，随着 $p$ 的增加，通信开销相对于每个进程的计算消耗不容忽视，加速比提升并不明显，而当 $n$ 较大且模拟的时间规模较大（即计算周期较多）时，较高的并行度可以减少小球速度与受力的计算，加速能有较好的效果。