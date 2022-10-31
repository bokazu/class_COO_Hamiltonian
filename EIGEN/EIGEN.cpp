#include "EIGEN.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>

#include "cblas.h"

using namespace std;

//コピーコンストラクタ
EIGEN::EIGEN(const EIGEN& x) : mat_dim(x.mat_dim), eigen_val(x.eigen_val)
{
    eigen_vec = new double[mat_dim];
    cblas_dcopy(mat_dim, x.eigen_vec, 1, eigen_vec, 1);
    std::cout << "copy has done\n";
}

//代入演算子
EIGEN& EIGEN::operator=(const EIGEN& x)
{
    cblas_dcopy(mat_dim, x.eigen_vec, 1, eigen_vec, 1);
    return *this;
}

//固有ベクトルの初期化(0で初期化する)
void EIGEN::vec_init()
{
    for (int i = 0; i < mat_dim; i++)
    {
        eigen_vec[i] = 0.0;
    }
}

//固有ベクトルの要素数を変更し、初期化する
void EIGEN::evec_elem(int num){
    delete[] eigen_vec;
    eigen_vec = new double[num];
    mat_dim = num;

    for(int i = 0; i < num; i++){
        eigen_vec[i] = 0.0;
    }
}

//文字列表現を返却する
std::string EIGEN::to_string() const
{
    std::ostringstream s;

    s << "@eigen_value = " << eigen_val << std::endl;
    s << "@eigen_vector  \n";
    for (int i = 0; i < mat_dim; i++)
    {
        s << "eigen_vec[" << i << "] = " << eigen_vec[i] << std::endl;
    }
    return s.str();
}

void EIGEN::to_file(std::string filename) const
{
    std::ofstream ofs(filename);

    ofs << "@eigen_value = " << eigen_val << std::endl;
    ofs << "@eigen_vector  \n";
    for (int i = 0; i < mat_dim; i++)
    {
        ofs << "eigen_vec[" << i << "] = " << eigen_vec[i] << std::endl;
    }
}

ostream& operator<<(ostream& s, const EIGEN& x) { return s << x.to_string(); }
