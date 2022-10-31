#ifndef ___Class_COO_Hamiltonian
#define ___Class_COO_Hamiltonian

#include <mkl.h>

#include <iomanip>
#include <string>

#include "../EIGEN/EIGEN.hpp"
#include "../Jset/Jset.hpp"

class COO_Hamiltonian
{
        private:
    std::string jset_filename;
    int tot_site_num;
    int mat_dim;
    int nnz;
    int* row;
    int* col;
    double* val;
    int ls_count;

    // Hamiltonian行列の非ゼロ要素の個数を数え上げる
    void count_nnz();
    // COO形式の配列の要素数を変更し、0で初期化する
    void set_CooElem(int i);
    // COO形式での行列ベクトル積の計算を行う(uをベクトル、HをHamiltonian行列としてu
    // += Huを計算する)
    void coo_mvprod(double* u_i, double* u_j);

    Jset J;
    void set_J() { J.set(); }
    void coo_spin(int m, int site_, int& coo_index, double& szz);

        public:
    EIGEN Eig;

    //コンストラクタ
    explicit COO_Hamiltonian(std::string filename, int site)
        : jset_filename(filename),
          tot_site_num(site),
          mat_dim(1 << tot_site_num),
          nnz(0),
          row(new int[1]),
          col(new int[1]),
          val(new double[1]),
          ls_count(0),
          J(filename),
          Eig(1)
    {
        set_J();
        std::cout << "CSC_Hamiltonian::constructed.\n";
    }

    //コピーコンストラクタ
    COO_Hamiltonian(const COO_Hamiltonian& h);

    //デストラクタ
    ~COO_Hamiltonian()
    {
        delete[] row;
        delete[] col;
        delete[] val;
        std::cout << "COO_Hamiltonian::destructed.\n";
    }

    //代入演算子
    COO_Hamiltonian& operator=(const COO_Hamiltonian& h);

    /*----------------------ゲッタ-----------------------*/
    // Jsetの情報を書き込んだファイルの名前を返す
    std::string jsetfile() const { return jset_filename; }

    //系のサイト数
    int site() const { return tot_site_num; }

    //行列の次元を返す
    int dim() const { return mat_dim; }

    // HamiltonianのNon zero要素の個数を返す
    int num_nnz() const { return nnz; }

    // Hamiltonianのrow[i]の値を返す
    int at_row(int i) const { return row[i]; }

    // Hamiltonianのcol[i]の値を返す
    int at_col(int i) const { return col[i]; }

    // Hamiltonianのval[i]の値を返す
    double at_val(int i) const { return val[i]; }

    // lanczos法での反復回数を返す
    int num_ls() const { return ls_count; }

    /*----------------------------------------------------------------*/

    /*------------------------その他メンバ関数------------------------*/
    // row, col, valの初期化をおこなう
    void init();

    //

    // COO形式でHamiltonian行列の行列要素の計算と配列への格納を行う
    void coo_hamiltonian();

    // lanczos法によりHamiltonianの基底状態の固有値、固有ベクトルを求める
    void coo_lanczos(int tri_mat_dim, char c = 'N', char info_ls = 'n');

    /*-------------------------------------------------------------------*/

    /*------------------------------標準出力関係-------------------------*/
    //標準出力において、倍精度を何桁まで出力するかを指定する
    int precision = 5;

    // precisionのsetter
    void set_precision(int i) { precision = i; }

    // row[i], col[i], val[i]を標準出力する
    void print(int i) const;

    // Hamiltonian行列の全要素を標準出力する
    void print() const;

    // COO_Hamiltonianオブジェクトの文字列表現を返却する
    std::string to_string() const;

    /*--------------------------------------------------------------------*/
};

//出力ストリームにhを挿入する
std::ostream& operator<<(std::ostream& s, const COO_Hamiltonian& h);

template <typename T>
void vec_init(int dim, T* vec)
{
    for (int i = 0; i < dim; i++)
    {
        vec[i] = 0;
    }
}

#endif