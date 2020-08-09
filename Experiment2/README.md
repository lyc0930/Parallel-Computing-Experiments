# <center>实验 2</center>

##### <p align="right">PB17000297 罗晏宸</br>2020.8.6</p>

### 实验题目

利用 MPI 进行蒙特卡洛模拟

在道路交通规划上，需要对单条道路的拥堵情况进行估计。因为仅考虑单条车道，所以不存在超车。假设共有 $n$ 辆车，分别编号 $0, 1, \cdots, n-1$ ，每辆车占据一个单位的空间。初始状态如下，$n$ 辆车首尾相连，速度都是  $0$ 。每个时间周期里每个车辆的运动满足以下规则：
1. 假设当前周期开始时，速度是 $v$ 。
2. 和前一辆车的距离为 $d$（前一辆车车尾到这辆车车头的距离，对于第 $0$ 号车，$d = \infty$），若 $d > v$，它的速度会提高到 $v + 1$ 。最高限速 $v_max$。若 $d \leqslant v$ ，那么它的速度会降低到 $d$。
3. 前两条完成后，司机还会以概率 $p$ 随机减速 $1$ 个单位。速度不会为负值。
4. 基于以上几点，车辆向前移动 $v$（这里的 $v$ 已经被更新）个单位

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

Monte Carlo 方法要求在每个时间周期内对每一辆车

1. 根据**其前方车辆的位置**以及速度，更新速度
2. 根据速度更新位置

而对于并行算法而言，均匀分配每个进程计算不同的车辆即可完成上述算法的并行化。注意到计算每个车辆的速度改变量需要其前一辆车的位置，因此为了减少通信量，考虑每个进程计算的车辆是连续的一段，在每个时刻计算前，向后一段发送队首车辆的位置信息，并接受前一段的队首车辆的位置信息。

需要指出的是，为了较为便利地模拟车辆移动且统一代码语言的表达，我们认为车辆行驶方向是车辆下标增加的方向，并在每一段中称最前方的车（下标最大）为队尾（`tail`），称最后方的车（下标最小）为队头（`head`），这相对是反直观的，特此指出。

按题目要求，在本问题中，考虑 $v_{\max} = 12$ ，$p = 0.2$

### 核心代码

#### 初始化

按问题描述，初始化车辆的位置与速度，并为每个进程分配车辆。

```c++
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
```

#### 周期计算

以时间为循环变量，每个时刻首先在两个计算相邻段的进程间通信队首信息，这里对消息标签的设计包含了当前周期，以保证消息的发送与接受不会跨周期。

```c++
    for (int i = 0; i < numberOfCycles; i++)
    {
        if (rank != 0) // 不为最后一部分
            MPI_Send(position + head, 1, MPI_UNSIGNED, rank - 1, i * (size - 1) + (rank - 1), MPI_COMM_WORLD);
        if (rank != size - 1) // 不为首个部分
        {
            MPI_Status status;
            MPI_Recv(position + tail, 1, MPI_UNSIGNED, rank + 1, i * (size - 1) + rank, MPI_COMM_WORLD, &status);
        }
```

之后对车辆的速度与位置进行计算

```c++
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
```

周期结束后，`root` 进程收集全局的速度与位置以便输出，这里 `root` 进程采用了`MPI_IN_PLACE`常量作为`MPI_Gatherv`的发送地址，表示数据同步后存放在源地址。

```c++
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
```

### 实验结果

- 编译运行
程序采用 MPICH 编译运行，在终端相应目录下执行

```shell
mpicxx -g -Wall Nagel-Schreckenberg.cpp -o Nagel-Schreckenberg.o
mpirun -n 4 ./Nagel-Schreckenberg.o 1000000 300
```

即以 4 个并行进程运行程序，数据规模为 $n = 1000000$ 周期数为 $300$

-  道路情况
每次程序运行都会在`traffic_condition`文件夹下输出文件名形如`NxC.data`的数据文件，其中`N`表示车辆数量，`C`表示周期数，文件中的数据条目以`Car i : Position , Velocity `的形式记录每辆车的位置与车速。

-  运行时间(s)

每个数据点的运行时间统计是 10 次运行的平均时间

| 规模（车辆数量 $\times$ 周期）\进程数 | 1           | 2           | 4           |
| :---------- | ----------- | ----------- | ----------- |
| 100000 $\times$ 2000        | 3.92441  | 2.82178  | 1.58254 |
| 500000 $\times$ 500       | 3.96222 | 2.73523 | 1.86402 |
| 1000000 $\times$ 300      | 5.35513  | 3.65961  | 2.31082  |

- 加速比

| 规模（车辆数量 $\times$ 周期）\进程数 | 1           | 2           | 4           |
| :---------- | ----------- | ----------- | ----------- |
| 100000 $\times$ 2000        | 1  | 1.39076  | 2.47982 |
| 500000 $\times$ 500       | 1 | 1.44859 | 2.12563 |
| 1000000 $\times$ 300      | 1  | 1.46331  | 2.31742 |

### 分析与总结

Monte Carlo 方法解决 Nagel Schreckenberg 交通模拟的算法是朴素的，主要的通信过程发生在每个时间周期内对车辆局部邻接处的位置信息交换。

另外注意到实验的数据点中，车辆数量都远大于周期数，这在题设情形下会导致只有队尾与周期数相当数量的车辆产生移动，而其余大量的车保持静止，对于并行的进程而言，即只有一个进程（可以认为是 `size - 1​` 号进程）的计算是有效的，因此实验中算法表现其实仅受周期数约束，并不能很好地表现出在车辆数量较大时的并行效果。