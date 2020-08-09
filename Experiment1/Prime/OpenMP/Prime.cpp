#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <omp.h>

using namespace std;

int main(int argc, char* argv[])
{
    const int numberOfThreads = atoi(argv[1]); // 获取并行数
    const int n = atoi(argv[2]);               // 获取级数规模

    double startTime = omp_get_wtime();

    bool isComposite[n + 1] = {true, true}; // 0 1 是合数

#pragma omp parallel for num_threads(numberOfThreads)
    for (int i = 2; i <= (int)sqrt(n); i++)
        if (!isComposite[i])
            for (int j = i; i * j <= n; j++)
                isComposite[i * j] = true;

    int total = 0;

#pragma omp parallel for reduction(+ : total) num_threads(numberOfThreads)
    for (int i = 3; i <= n; i += 2)
        if (!isComposite[i])
            total++;

    if (n >= 2)
        total++;

    double endTime = omp_get_wtime();

    cout << "The number of prime numbers is " << total << endl;
    cout << "Elapsed time = " << 1E6 * (endTime - startTime) << "microsecond" << endl;
    return 0;
}