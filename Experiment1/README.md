# <center>实验 1</center>

##### <p align="right">PB17000297 罗晏宸</br>2020.7.7</p>

### 实验题目

利用 MPI，OpenMP 编写简单的程序，测试并行计算系统性能

- 求素数个数
- 求 $\pi$ 值

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

## <center>求素数个数</center>

给定正整数 $n$ ，编写程序计算出所有小于等于 $n$ 的素数的个数

### 算法设计与分析

首先通过筛法维护对素数的判定数组，再通过累计求和计算素数的个数。其中并行优化累计求和的部分，通过基本的归约操作实现求和任务的分配。

### <center>MPI 版本</center>

### 核心代码

代码学习了《并行计算——结构·算法·编程》 P425 15.2.6

为提高程序的紧凑程度，用对合数的判定数组 `bool isComposite[n + 1]` 代替对素数的判定数组，用筛法对其进行维护

```c++
    for (int i = 2; i <= (int)sqrt(n); i++)
        if (!isComposite[i])
            for (int j = i; i * j <= n; j++)
                isComposite[i * j] = true;
```

向每个进程分配奇数进行素数的计数，注意到这里的循环实质上完成了对于奇数的间断分配。

```c++
    int local = 0, total;

    for (int i = 2 * rank + 1; i <= n; i += 2 * size)
        if (!isComposite[i])
            local++;
```

通过以 `MPI_SUM` 为参数的 `MPI_Reduce` 对每个进程中的局部数据进行归约求和

```c++
    MPI_Reduce(&local, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // NOTE:
    // 归约 Reduce 操作对每个进程的发送缓冲区 &local 中的数据按给定的操作进行运算，
    // 并将最终结果存放在 0 进程的接受缓冲区 &total 中。
    // 参与计算操作的数据项的数据类型为 MPI_INT ，数据（向量）长度为 1 ，
    // 归约操作为 MPI_SUM 即求和

    if (n >= 2)
        total++;
```

### 实验结果

- 编译运行
程序采用 MPICH 编译运行，在终端相应目录下执行

```shell
mpicxx -g -Wall -o Prime.o Prime.cpp
mpirun -n 4 ./Prime.o 1000
```

即以 4 个并行进程运行程序，数据规模为 $n = 1000$

-  运行时间($\mu$s)

每个数据点的运行时间统计是 10 次运行的平均时间

| 规模\进程数 | 1           | 2           | 4           | 8         |
| :---------- | ----------- | ----------- | ----------- | --------- |
| 1000        | 46.4916  | 19.9097 | 16.2326 | 1534.87 |
| 10000       | 354.115 | 242.335 | 184.510 | 45875.3 |
| 100000      | 7048.12  | 4073.26  | 4835.16  | 92092.8 |
| 500000      | 98943.0   | 54282.9  | 60606.0  | 116115.0 |

- 加速比

| 规模\进程数 | 1    | 2       | 4       | 8          |
| ----------- | ---- | ------- | ------- | ---------- |
| 1000        | 1    | 2.33512 | 2.86408 | 0.0302903  |
| 10000       | 1    | 1.46126 | 1.91922 | 0.00771908 |
| 100000      | 1    | 1.73034 | 1.43768 | 0.0765328  |
| 500000      | 1    | 1.82273 | 1.63256 | 0.852115   |

### <center>OpenMP 版本</center>

### 核心代码

代码学习了《并行计算——结构·算法·编程》 P400 14.3.5。

与基于 MPI 的代码基本一致，用筛法对对合数的判定数组 `bool isComposite[n + 1]` 进行维护

```c++
#pragma omp parallel for num_threads(numberOfThreads)
    for (int i = 2; i <= (int)sqrt(n); i++)
        if (!isComposite[i])
            for (int j = i; i * j <= n; j++)
                isComposite[i * j] = true;
```

向每个进程分配奇数进行素数的计数，并利用 `reduction(+ : total)` 数据域属性子句对每个进程中的局部数据进行归约求和

```c++
    int total = 0;

#pragma omp parallel for reduction(+ : total) num_threads(numberOfThreads)
    // NOTE:
    // #pragma omp 编译制导指令前缀
    // parallel for OpenMP 制导指令，创建一个包含一个单独 for 语句的并行域
    //     由于没有指定 schedule(type[, chunk]) ， for 迭代会尽可能平均地分配给各线程
    // reduction(+ : total) 数据域属性子句，使用操作 + 对列表中出现的变量 total 进行归约
    //     初始时，对列表中的每个变量，线程组中的每个线程都将会保留一个私有副本。
    //     在并行结构尾部，根据指定操作对所有线程中的相应变量进行归约，并更新全局值
    // num_threads(numberOfThreads) 指定并行域的线程数
    for (int i = 3; i <= n; i += 2)
        if (!isComposite[i])
            total++;

    if (n >= 2)
        total++;
```

### 实验结果

- 编译运行
程序采用 g++ & OpenMP 编译运行，在终端相应目录下执行

```shell
g++ -O3 -fopenmp Prime.cpp -o Prime.o -Wall -g
./Prime.o 4 1000
```

即以 4 个并行进程运行程序，数据规模为 $n = 1000$

-  运行时间($\mu$s)

每个数据点的运行时间统计是 10 次运行的平均时间

> 基于 OpenMP 的程序运行时间由 `omp_get_wtime()` 统计

| 规模\进程数 | 1          | 2          | 4           | 8          |
| :---------- | ---------- | ---------- | ----------- | ---------- |
| 1000        | 26.6  | 16.8994 | 13.0259 | 1738.9  |
| 10000       | 55.9  | 39.3701 | 28.6678 | 1431.72  |
| 100000      | 227.5 | 149.28 | 89.9875 | 6798.96 |
| 500000      | 1468.8  | 902.565 | 676.071 | 14083.5 |


- 加速比

| 规模\进程数 | 1    | 2       | 4       | 8          |
| ----------- | ---- | ------- | ------- | ---------- |
| 1000        | 1    | 1.57402 | 2.04208 | 0.015297  |
| 10000       | 1    | 1.41986 | 1.94992 | 0.039044 |
| 100000      | 1    | 1.52398 | 2.52813 | 0.033461  |
| 500000      | 1    | 1.62703 | 2.17211 | 0.104271   |


## <center>求 $\pi$ 值</center>

### 算法设计与分析

求 $\pi$ 值的算法基于公式

$$
    \pi = \int_{0}^{1}\!{\frac{4}{1 + x^2}\,\text{d}x}
$$

定积分可由复化积分近似计算，考虑 $(0, 1)$ 上的 $\dfrac{1}{2n}, \dfrac{3}{2n}, \cdots, \dfrac{2n - 1}{2n}$ 共 $n$ 个点，每个点 $x_i = \dfrac{2i + 1}{2n},\ i = 0, \cdots, n - 1$ 处的被积函数值为 $\displaystyle \frac{4}{1 + x_i^2}$ ，以之为高，以 $\dfrac{1}{n}$ 为宽计算矩形面积并求和，即可得到数值积分值。

通过基本的归约操作即可完成对于矩形面积计算任务的分配。

### <center>MPI 版本</center>

### 核心代码

对每一个进程，间断分配约 $\dfrac{n}{p}$ 个点

```c++
    double local = 0, pi;
    for (int i = rank; i < n; i += size)
    {
        double x = (i + 0.5) / n;
        local += 4.0 / (1.0 + pow(x, 2));
    }
```

通过归约操作对所有矩形面积求和

```c++
    MPI_Reduce(&local, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // NOTE:
    // 归约 Reduce 操作对每个进程的发送缓冲区 &local 中的数据按给定的操作进行运算，
    // 并将最终结果存放在 0 进程的接受缓冲区 &pi 中。
    // 参与计算操作的数据项的数据类型为 MPI_DOUBLE ，数据（向量）长度为 1 ，
    // 归约操作为 MPI_SUM 即求和

    pi /= n;
```


### 实验结果

- 编译运行
程序采用 MPICH 编译运行，在终端相应目录下执行

```shell
mpicxx -g -Wall -o pi.o pi.cpp
mpirun -n 4 ./pi.o 10000
```

即以 4 个并行进程运行程序，数据规模为 $n = 10000$

-  运行时间($\mu$s)

每个数据点的运行时间统计是 10 次运行的平均时间

| 规模\进程数 | 1           | 2           | 4           | 8         |
| :---------- | ----------- | ----------- | ----------- | --------- |
| 1000        | 164.133  | 120.561 | 55.9976 | 2762.8 |
| 10000       | 223.241 | 197.662 | 115.824 | 200684.0 |
| 50000       | 2675.77  | 1747.49 | 1082.6 | 70078.6 |
| 100000      | 13824.2   | 10081.8 | 4358.05 | 71824.7 |

- 加速比

| 规模\进程数 | 1    | 2       | 4       | 8          |
| ----------- | ---- | ------- | ------- | ---------- |
| 1000        | 1    | 1.36141 | 2.93107 | 0.0594082  |
| 10000       | 1    | 1.12941 | 1.92741 | 0.0011124  |
| 50000       | 1    | 1.53121 | 2.47162 | 0.0381824  |
| 100000      | 1    | 1.37121 | 3.17211 | 0.1924715  |

### <center>OpenMP 版本</center>

### 核心代码

利用 `parallel for` 制导命令以及 `reduction(+ : pi)` 数据域属性子句直接完成计算任务的分配与归约

```c++
#pragma omp parallel for schedule(guided) reduction(+ : pi) num_threads(numberOfThreads)
    // NOTE:
    // #pragma omp 编译指导指令前缀
    // parallel for OpenMP 制导指令，创建一个包含一个单独 for 语句的并行域
    //     指定 schedule(guided) ， for 迭代自动分配给各线程
    // reduction(+ : pi) 数据域属性子句，使用操作 + 对列表中出现的变量 pi 进行归约
    //     初始时，对列表中的每个变量，线程组中的每个线程都将会保留一个私有副本。
    //     在并行结构尾部，根据指定操作对所有线程中的相应变量进行归约，并更新全局值
    // num_threads(numberOfThreads) 指定并行域的线程数

    for (int i = 0; i < n; i++)
    {
        double x = (i + 0.5) / n;
        pi += 4.0 / (1.0 + pow(x, 2));
    }

    pi /= n;
```

### 实验结果

- 编译运行
程序采用 g++ & OpenMP 编译运行，在终端相应目录下执行

```shell
g++ -O3 -fopenmp pi.cpp -o pi.o -Wall -g
./pi.o 4 10000
```

即以 4 个并行进程运行程序，数据规模为 $n = 10000$

-  运行时间($\mu$s)

每个数据点的运行时间统计是 10 次运行的平均时间

> 基于 OpenMP 的程序运行时间由 `omp_get_wtime()` 统计

| 规模\进程数 | 1          | 2          | 4           | 8          |
| :---------- | ---------- | ---------- | ----------- | ---------- |
| 1000        |   162.7  |  97.59  |  44.29   |  671.8  |
| 10000       |   427.9  |  379.7  |  120.2   |  733.9  |
| 100000      |  2929.69 | 1732.28 |  876.832 | 23952.0 |
| 500000      | 14648.4  | 8407.46 | 4413.47  | 20604.8 |


- 加速比

| 规模\进程数 | 1    | 2       | 4       | 8          |
| ----------- | ---- | ------- | ------- | ---------- |
| 1000        | 1    | 1.66718 | 3.67352 | 0.242185  |
| 10000       | 1    | 1.12694 | 3.55992 | 0.583049 |
| 100000      | 1    | 1.69123 | 3.34122 | 1.50001  |
| 500000      | 1    | 1.74231 | 3.31902 | 0.710921   |

### 分析与总结

观察各组运行结果，随着并行数的增加，运行时间通常有所减少，加速比增加，但可以看到当进程数超过逻辑 CPU 核数（即线程数 4）时，运行时间显著增加，这是由于每个逻辑 CPU 核心同一时间内最多只能运行一个进程，当并行进程数超过线程数时，CPU 会进行多次进程间的切换调整多个进程的并行运行情况，而进程切换会消耗额外的时间。同时对于逻辑 CPU 核数以内的进程数，加速比也并不总是显著的增加，这受进程间通信开销的影响。

以求 $\pi$ 值的实验为例，每个进程的串行计算部分消耗其实非常之小，相比之下并行的通信消耗就会显著影响程序执行时间。另外注意到计算素数个数的程序中，对于各进程的任务分配是平均的，但实际上素数的分布并不平均，这启示并行的算法表现同样受数据分布的影响。

### 注：关于本课程实验的统一说明

#### MPI 并行框架

在本课程全部 4 次实验的代码中，每个基于 MPI 的程序代码的主程序通常形如

```c++
int main(int argc, char** argv)
{
    int rank, size;
    const int n = atoi(argv[1]);
    // other arguments

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double startTime = MPI_Wtime();

    // main code

    double endTime = MPI_Wtime();

    if (rank == 0)
    {
        // output

        cout << "Elapsed time = " << 1E6 * (endTime - startTime) << "microsecond" << endl;
    }

    MPI_Finalize();
    return 0;
}
```

其中 `rank` 表示本地进程编号， `size` 表示总进程数目， MPI 的初始化参数均为 `NULL` 。由命令行输入 `argv` 得到程序所需的参数。程序的运行时间由 `MPI_Wtime()` 统计，并且不包括初始化与输出的时间，且输出的时间单位可能有变动。

这部分内容在之后的实验报告中不再赘述。

#### 并行进程数的选择

如上所述，受硬件条件限制，当进程数超过逻辑 CPU 核数（即线程数 4）时，运行时间显著增加，这是由于每个逻辑 CPU 核心同一时间内最多只能运行一个进程，当并行进程数超过线程数时，CPU 会进行多次进程间的切换调整多个进程的并行运行情况，而进程切换会消耗额外的时间。在此后的实验中，进程数的选取限于1,2,4，在之后的实验报告中不再赘述。