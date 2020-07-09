// 《并行计算——结构·算法·编程》 P400 14.3.5
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <omp.h>

using namespace std;

int main(int argc, char **argv)
{
    double pi = 0;
    const int numberOfThreads = atoi(argv[1]); // 获取并行数
    const int n = atoi(argv[2]);               // 获取级数规模

    double startTime = omp_get_wtime();

#pragma omp parallel for schedule(guided) reduction(+ : pi) num_threads(numberOfThreads)
    // NOTE:
    // #pragma omp 编译指导指令前缀
    // parallel for OpenMP 指导指令，创建一个包含一个单独 for 语句的并行域
    //     由于没有指定 schedule(type[, chunk]) ， for 迭代会尽可能平均地分配给各线程
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
    cout << "pi = " << pi << endl;

    double endTime = omp_get_wtime();

    cout << "Elapsed time = " << endTime - startTime << "s" << endl;
}