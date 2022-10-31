#ifndef ___Class_EIGEN
#define ___Class_EIGEN

#include <mkl.h>

#include <iomanip>
#include <iostream>

class EIGEN
{
        private:
    int mat_dim;
    double eigen_val;
    double* eigen_vec;

        public:
    //コンストラクタ
    EIGEN(int dim) : mat_dim(dim), eigen_val(0)
    {
        eigen_vec = new double[mat_dim];
        for (int i = 0; i < mat_dim; i++)
        {
            eigen_vec[i] = 0.0;
        }
        std::cout << "EIGEN::constructed" << std::endl;
    }
    //コピーコンストラクタ
    EIGEN(const EIGEN& eigen);

    //デストラクタ
    ~EIGEN()
    {
        delete[] eigen_vec;
        std::cout << "EIGEN::destructed" << std::endl;
    }

    //代入演算子=
    EIGEN& operator=(const EIGEN& x);

    //添字演算子[]
    double& operator[](int i) { return eigen_vec[i]; }

    // const版添字演算子[]
    const double& operator[](int i) const { return eigen_vec[i]; }

    //固有値をセットする
    void set_eval(double val) { eigen_val = val; }

    //固有値の値を取得する
    double eval() const { return eigen_val; }

    //固有ベクトルの第[i]成分を返却する
    double evec(int i) const { return eigen_vec[i]; }

    double* data() { return eigen_vec; }
    //固有ベクトルの値を初期化する
    void vec_init();

    //固有ベクトルの要素数を変更し、初期化する
    void evec_elem(int i);

    //固有値、固有ベクトルの文字列表現を返却する
    std::string to_string() const;
    void to_file(std::string filename) const;
};

//出力ストリームにxを挿入する
std::ostream& operator<<(std::ostream& s, const EIGEN& x);

#endif
