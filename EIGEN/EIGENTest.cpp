#include <iomanip>
#include <iostream>

#include "EIGEN.hpp"
#include "cblas.h"

using namespace std;

int main()
{
    int mat_dim = 4;
    EIGEN E1(mat_dim);
    for (int i = 0; i < mat_dim; i++)
    {
        E1[i] = i * 2.0;
    }
    E1.set_eval(2.0);

    for (int i = 0; i < mat_dim; i++)
    {
        cout << "E1's eigen_vec[" << i << "] = " << E1.evec(i) << endl;
    }

    cout << E1 << endl;

    E1.to_file("EIGENTest.txt");

    EIGEN E2 = E1;

    E2.vec_init();
    cout << E2 << endl;

    E2.evec_elem(10);
    cout << E2 << endl;
}
