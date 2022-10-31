#include "COO_Hamiltonian/COO_Hamiltonian.hpp"

using namespace std;

int main()
{
    COO_Hamiltonian H("jset_16site_pb.txt", 16);
    H.coo_hamiltonian();

    H.coo_lanczos(300);

    cout << H << endl;
}