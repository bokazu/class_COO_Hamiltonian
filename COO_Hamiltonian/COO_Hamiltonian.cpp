#include "COO_Hamiltonian.hpp"

#include <mkl.h>

#include <boost/dynamic_bitset.hpp>
#include <fstream>
#include <iomanip>
#include <random>

using namespace std;

//コピーコンストラクタ
COO_Hamiltonian::COO_Hamiltonian(const COO_Hamiltonian &h)
    : jset_filename(h.jset_filename),
      tot_site_num(h.tot_site_num),
      mat_dim(h.mat_dim),
      nnz(h.nnz),
      J(h.J),
      Eig(h.Eig)
{
    row = new int[nnz];
    col = new int[nnz];
    val = new double[nnz];

    for (int i = 0; i < nnz; i++)
    {
        row[i] = h.row[i];
        col[i] = h.col[i];
    }
    cblas_dcopy(nnz, h.val, 1, val, 1);
    std::cout << "CSC_Hamiltonian::copy_has_done\n";
}

//代入演算子
COO_Hamiltonian &COO_Hamiltonian::operator=(const COO_Hamiltonian &h)
{
    if (h.nnz != nnz)
    {
        std::cout << "COO_Hamiltonian::bad array. Please set same nnz array "
                     "both side of \"=\" . \n";
    }
    else
    {
        for (int i = 0; i < nnz; i++)
        {
            row[i] = h.row[i];
            col[i] = h.col[i];
        }
        cblas_dcopy(nnz, h.val, 1, val, 1);
    }

    return *this;
}

// row, col,valの初期化を行う
void COO_Hamiltonian::init()
{
    for (int i = 0; i < nnz; i++)
    {
        row[i] = 0;
        col[i] = 0;
        val[i] = 0.0;
    }
}

// COO形式の配列の要素数を変更し、すべて0で初期化する
void COO_Hamiltonian::set_CooElem(int i)
{
    delete[] row;
    delete[] col;
    delete[] val;

    row = new int[i];
    col = new int[i];
    val = new double[i];

    this->init();
}

//  Hamiltonian行列の非ゼロ要素の個数をカウントする
void COO_Hamiltonian::count_nnz()
{
    nnz = 0;

    double szz;

    for (int m = 0; m < mat_dim; m++)
    {
        szz = 0.;
        for (int site_i = 0; site_i < tot_site_num; site_i++)
        {
            boost::dynamic_bitset<> ket_m(tot_site_num, m);
            bool bit_check0, bit_check1;

            int j_line = 0;

            for (int l = j_line; l < J.get_line(); l++)
            {
                if (J.index(0, l) == site_i)
                {
                    int site_j = J.index(1, l);
                    bit_check0 = ket_m.test(site_i);
                    bit_check1 = ket_m.test(site_j);

                    //注目する2サイトのスピン状態についての場合分け
                    if (bit_check0 == bit_check1)
                    {
                        szz += 0.25 * J.val(l);
                    }
                    else
                    {
                        boost::dynamic_bitset<> ket_m1(tot_site_num, m);

                        ket_m1.flip(site_j);
                        ket_m1.flip(site_i);
                        int n = ket_m1.to_ulong();

                        nnz++;
                        szz -= 0.25 * J.val(l);
                    }
                }
            }
        }
        if (szz != 0.) nnz++;
    }
}

// spin演算子と状態ベクトルの計算を行う
void COO_Hamiltonian::coo_spin(int m, int site_i, int &coo_index, double &szz)
{
    boost::dynamic_bitset<> ket_m(tot_site_num, m);
    bool bit_check0, bit_check1;

    int j_line = 0;

    for (int l = j_line; l < J.get_line(); l++)
    {
        if (J.index(0, l) == site_i)
        {
            int site_j = J.index(1, l);
            bit_check0 = ket_m.test(site_i);
            bit_check1 = ket_m.test(site_j);

            //注目する2サイトのスピン状態についての場合分け
            if (bit_check0 == bit_check1)
            {
                szz += 0.25 * J.val(l);
            }
            else
            {
                boost::dynamic_bitset<> ket_m1(tot_site_num, m);

                ket_m1.flip(site_j);
                ket_m1.flip(site_i);
                int n = ket_m1.to_ulong();

                row[coo_index] = n;
                col[coo_index] = m;
                val[coo_index] = 0.5 * J.val(l);
                coo_index++;

                szz -= 0.25 * J.val(l);
            }
        }
    }
}

// Hamiltonian行列要素の計算とCOO形式での配列への格納を行う
void COO_Hamiltonian::coo_hamiltonian()
{
    count_nnz();
    // COO形式の各種配列のメモリを再確保する(デフォルトは要素1つしか確保していないため)
    set_CooElem(nnz);

    int coo_index = 0;
    double szz;
    for (int m = 0; m < mat_dim; m++)
    {
        szz = 0.;
        for (int site_i = 0; site_i < tot_site_num; site_i++)
        {
            coo_spin(m, site_i, coo_index, szz);
        }

        if (szz != 0.0)
        {
            row[coo_index] = m;
            col[coo_index] = m;
            val[coo_index] = szz;

            coo_index++;
        }
    }
};

/*============================Lanczos
 * Algorithm関係の関数=================================*/

//ベクトルの規格化用の関数(非メンバ関数)
void sdz(int dim, double *vec)
{
    double a = 1. / cblas_dnrm2(dim, vec, 1);
    cblas_dscal(dim, a, vec, 1);
}

// COO形式での行列ベクトル積の計算を行う(uをベクトル、HをHamiltonian行列としてu_j
// += Hu_iを計算する)
void COO_Hamiltonian::coo_mvprod(double *u_i, double *u_j)
{
    for (int i = 0; i < nnz; i++)
    {
        u_j[row[i]] += val[i] * u_i[col[i]];
    }
}

// Hamiltonianの基底状態の固有値と固有ベクトルを計算する
void COO_Hamiltonian::coo_lanczos(int tri_mat_dim, char c, char info_ls)
{
    ls_count = 0;
    double eps = 1.0;      // step間の誤差を代入するための変数
    double err = 1.0e-16;  //要求精度
    bool err_checker =
        true;  //誤差が要求精度の範囲内に収まっているかを確認するためのflag

    //固有ベクトルを格納するための配列の要素数を変更する(デフォルトは要素数1)
    if (c == 'V') Eig.evec_elem(mat_dim);

    /*=================初期ベクトルの用意================*/
    double **u = new double *[2];
    for (int k = 0; k < 2; k++)
    {
        u[k] = new double[mat_dim];
    }

    random_device rand;
    mt19937 mt(rand());
    uniform_real_distribution<> rand1(0, 1);
    for (int k = 0; k < mat_dim; k++)
    {
        u[0][k] = rand1(mt);
        u[1][k] = 0.0;
    }
    sdz(mat_dim, u[0]);

    if (c == 'V') cblas_dcopy(mat_dim, u[0], 1, Eig.data(), 1);
    /*============================================*/

    //三重対角行列の主対角成分
    double *alpha = new double[tri_mat_dim];
    vec_init(tri_mat_dim, alpha);

    //三重対角行列の次対角成分
    double *beta = new double[tri_mat_dim - 1];
    vec_init(tri_mat_dim - 1, beta);

    // ls = 偶数stepでの近似固有値
    double *eval_even = new double[tri_mat_dim];
    vec_init(tri_mat_dim, eval_even);

    // ls = 奇数stepでの近似固有値
    double *eval_odd = new double[tri_mat_dim];
    vec_init(tri_mat_dim, eval_odd);

    // LAPACKに三重対角行列の主対角成分を渡す用の配列
    double *diag = new double[tri_mat_dim];
    vec_init(tri_mat_dim, diag);

    // LAPACKに三重対角行列の主対角成分を渡す用の配列
    double *sub_diag = new double[tri_mat_dim - 1];
    vec_init(tri_mat_dim - 1, sub_diag);

    // LAPACKに渡し、c = 'N'なら参照されず、'V'なら固有ベクトルが格納される
    double *tri_diag_evec;

    //固有ベクトルを計算する場合は配列を確保する
    if (c == 'V')
    {
        tri_diag_evec = new double[tri_mat_dim];
        vec_init(tri_mat_dim, tri_diag_evec);
    }

    /*==============================lanczos
     * Algorithm=======================================*/
    for (int ls = 0; ls < tri_mat_dim; ls++)
    {
        if (err_checker)
        {
            ls_count = ls;
            /*省メモリのためのlanczosベクトルの更新*/
            if (ls > 0)
            {
                if (ls % 2 == 0)
                    cblas_dscal(mat_dim, -beta[ls - 1], u[1], 1);
                else
                    cblas_dscal(mat_dim, -beta[ls - 1], u[0], 1);
            }

            if (ls % 2 == 0)
            {
                // ls = 偶数step
                coo_mvprod(u[0], u[1]);
                alpha[ls] = cblas_ddot(mat_dim, u[1], 1, u[0], 1);
                cblas_daxpy(mat_dim, -alpha[ls], u[0], 1, u[1], 1);
                if (ls != tri_mat_dim - 1)
                {
                    beta[ls] = cblas_dnrm2(mat_dim, u[1], 1);
                    cblas_dscal(mat_dim, 1. / beta[ls], u[1], 1);
                }
            }
            else
            {
                // ls = 奇数step
                coo_mvprod(u[1], u[0]);
                alpha[ls] = cblas_ddot(mat_dim, u[1], 1, u[0], 1);
                cblas_daxpy(mat_dim, -alpha[ls], u[1], 1, u[0], 1);
                if (ls != tri_mat_dim - 1)
                {
                    beta[ls] = cblas_dnrm2(mat_dim, u[0], 1);
                    cblas_dscal(mat_dim, 1. / beta[ls], u[0], 1);
                }
            }
            /*===================================================================================*/

            /*===========================三重対角行列の数値対角(LAPACK)=============================*/
            vec_init(tri_mat_dim, diag);
            vec_init(tri_mat_dim - 1, sub_diag);
            int info = 0;
            if (ls % 2 == 0)
            {
                //偶数step
                cblas_dcopy(tri_mat_dim, alpha, 1, diag, 1);
                cblas_dcopy(tri_mat_dim - 1, beta, 1, sub_diag, 1);

                if (ls < tri_mat_dim - 1)
                {
                    sub_diag[ls] = 0.;
                    if (c == 'N')
                    {
                        //固有値のみを計算する場合
                        info =
                            LAPACKE_dstev(LAPACK_COL_MAJOR, 'N', ls + 1, diag,
                                          sub_diag, tri_diag_evec, ls + 1);
                    }
                    else
                    {  //固有ベクトルも計算する場合
                        info =
                            LAPACKE_dstev(LAPACK_COL_MAJOR, 'V', ls + 1, diag,
                                          sub_diag, tri_diag_evec, ls + 1);
                    }

                    if (info != 0)
                    {
                        std::cout << "@ls = " << ls
                                  << " , LAPACKE_detev errored." << std::endl;
                    }
                }
                else
                {
                    if (c == 'N')
                    {
                        //固有値のみを計算する場合
                        info =
                            LAPACKE_dstev(LAPACK_COL_MAJOR, 'N', ls + 1, diag,
                                          sub_diag, tri_diag_evec, ls + 1);
                    }
                    else
                    {
                        //固有ベクトルを計算する場合
                        info =
                            LAPACKE_dstev(LAPACK_COL_MAJOR, 'V', ls + 1, diag,
                                          sub_diag, tri_diag_evec, ls + 1);
                    }

                    if (info != 0)
                    {
                        std::cout << "@ls = " << ls
                                  << " , LAPACKE_detev errored." << std::endl;
                    }
                }
                cblas_dcopy(tri_mat_dim, diag, 1, eval_even, 1);
                if (info_ls == 'y')
                {
                    std::cout << "@ls = " << ls
                              << " : eigen value = " << eval_even[0]
                              << std::endl;
                }
                else if (info_ls == 's')
                {
                    std::cout << "@ls = " << ls << std::endl;
                }
            }
            else
            {
                cblas_dcopy(tri_mat_dim, alpha, 1, diag, 1);
                cblas_dcopy(tri_mat_dim - 1, beta, 1, sub_diag, 1);

                if (ls < tri_mat_dim - 1)
                {
                    sub_diag[ls] = 0.;
                    if (c == 'N')
                    {
                        //固有値のみを計算する場合
                        info =
                            LAPACKE_dstev(LAPACK_COL_MAJOR, 'N', ls + 1, diag,
                                          sub_diag, tri_diag_evec, ls + 1);
                    }
                    else
                    {
                        //固有ベクトルのみを計算する場合
                        info =
                            LAPACKE_dstev(LAPACK_COL_MAJOR, 'V', ls + 1, diag,
                                          sub_diag, tri_diag_evec, ls + 1);
                    }

                    if (info != 0)
                    {
                        std::cout << "@ls = " << ls
                                  << " , LAPACKE_detev's error." << std::endl;
                    }
                }
                else
                {
                    int info =
                        LAPACKE_dstev(LAPACK_COL_MAJOR, 'N', ls + 1, diag,
                                      sub_diag, tri_diag_evec, ls + 1);
                    if (info != 0)
                    {
                        std::cout << "@ls = " << ls
                                  << " , LAPACKE_detev's error." << std::endl;
                    }
                }
                cblas_dcopy(tri_mat_dim, diag, 1, eval_odd, 1);
                if (info_ls == 'y')
                {
                    std::cout << "@ls = " << ls
                              << " : eigen value = " << eval_odd[0]
                              << std::endl;
                }
                else if (info_ls == 's')
                {
                    cout << "@ls = " << ls << endl;
                }
            }
            /*======================================================================*/

            /*============================収束状況の確認==============================*/
            if (ls > 0)
            {
                eps = abs(eval_even[0] - eval_odd[0]);
                if (info_ls == 'y')
                {
                    cout << "eps = " << std::setprecision(16) << eps << endl;
                }

                if (eps > err)
                    err_checker = true;
                else
                    err_checker = false;
            }
            /*=====================================================================*/
        }
    }

    /*========================基底状態の固有値===========================*/
    if (ls_count % 2 == 0)
        Eig.set_eval(eval_even[0]);
    else
        Eig.set_eval(eval_odd[0]);
    /*========================配列リソースのリリース part1===================*/
    delete[] eval_even;
    delete[] eval_odd;

    //固有ベクトルを計算する
    if (c == 'V')
    {
        vec_init(mat_dim, u[0]);
        vec_init(mat_dim, u[1]);
        cblas_dcopy(mat_dim, Eig.data(), 1, u[0], 1);

        for (int ls = 0; ls < ls_count + 2; ls++)
        {
            if (ls % 2 == 0)
            {
                if (ls == 0)
                    cblas_dscal(mat_dim, tri_diag_evec[ls], Eig.data(), 1);
                else
                    cblas_daxpy(mat_dim, tri_diag_evec[ls], u[0], 1, Eig.data(),
                                1);
            }
            else
            {
                cblas_daxpy(mat_dim, tri_diag_evec[ls], u[1], 1, Eig.data(), 1);
            }

            if (ls > 0)
            {
                if (ls % 2 == 0)
                    cblas_dscal(mat_dim, -beta[ls - 1], u[1], 1);
                else
                    cblas_dscal(mat_dim, -beta[ls - 1], u[0], 1);
            }

            if (ls % 2 == 0)
            {
                coo_mvprod(u[0], u[1]);
                cblas_daxpy(mat_dim, -alpha[ls], u[0], 1, u[1], 1);
                if (ls != tri_mat_dim - 1)
                    cblas_dscal(mat_dim, 1. / beta[ls], u[1], 1);
            }
            else
            {
                coo_mvprod(u[1], u[0]);
                cblas_daxpy(mat_dim, -alpha[ls], u[1], 1, u[0], 1);
                if (ls != tri_mat_dim - 1)
                    cblas_dscal(mat_dim, 1. / beta[ls], u[0], 1);
            }
        }
        sdz(mat_dim, Eig.data());
    }

    /*==========================配列リソースのリリース part2
     * ====================*/
    for (int i = 0; i < 2; i++)
    {
        delete[] u[i];
    }
    delete[] u;
    delete[] alpha;
    delete[] beta;

    delete[] diag;
    delete[] sub_diag;
    if (c == 'V') delete[] tri_diag_evec;
}

/*========================================================================================*/
// row,col,valの第i成分を出力する
void COO_Hamiltonian::print(int i) const
{
    if (nnz < i)
    {
        std::cout << "index error@print(int i): nnz < i.\n ";
    }
    else
    {
        std::cout << "==========================================\n";
        std::cout << " row[" << i << "]"
                  << " "
                  << " col[" << i << "]"
                  << " "
                  << " val[" << i << "]";
        std::cout << "------------------------------------------\n";
        std::cout << row[i] << " " << col[i] << " " << setprecision(precision)
                  << val[i] << std::endl;
        std::cout << "==========================================\n";
    }
}

// row, col, valの全成分を出力する
void COO_Hamiltonian::print() const
{
    std::cout << "==========================================\n";
    std::cout << setw(5) << "row" << setw(5) << "col" << setw(5 + precision)
              << setprecision(precision) << " val\n";
    std::cout << "------------------------------------------\n";
    for (int i = 0; i < nnz; i++)
    {
        std::cout << setw(5) << row[i] << setw(5) << col[i]
                  << setw(5 + precision) << val[i] << std::endl;
    }
    std::cout << "==========================================\n";
}

//文字列表現を返却する
std::string COO_Hamiltonian::to_string() const
{
    std::ostringstream s;

    s << "Information of Hamiltonian\n";
    s << "-------------------------------------------------------\n";
    s << "@Number of site   : " << tot_site_num << std::endl;
    s << "@Matrix dimension : " << mat_dim << std::endl;
    s << "@non zero element : " << nnz << std::endl;
    s << J.to_string() << std::endl;
    s << "@lanczos step     : " << ls_count << std::endl;
    s << Eig.to_string() << std::endl;

    return s.str();
}

ostream &operator<<(ostream &s, const COO_Hamiltonian &h)
{
    return s << h.to_string();
}