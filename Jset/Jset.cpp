#include "Jset.hpp"

#include <mkl.h>

#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

//コピーコンストラクタ
Jset::Jset(const Jset& x)
{
    if (&x == this)
    {
        J_index = NULL;
        J_val = NULL;
    }
    else
    {
        jset_filename = x.jset_filename;
        jset_line = x.jset_line;
        J_index = new int*[2];
        for (int i = 0; i < 2; i++) J_index[i] = new int[jset_line];
        J_val = new double[jset_line];
        // cblasのdccopyを使用して要素を代入する
        for (int i = 0; i < jset_line; i++)
        {
            J_index[0][i] = x.J_index[0][i];
            J_index[1][i] = x.J_index[1][i];
        }
        cblas_dcopy(jset_line, x.J_val, 1, J_val, 1);
    }
    cout << "Jset::copy constructer was called.\n";
}

// //代入演算子
Jset& Jset::operator=(const Jset& x)
{
    if (&x != this)
    {
        if (jset_line != x.jset_line)
            cout << "Jset::bad_array\n";
        else
        {
            jset_filename = x.jset_filename;
            for (int i = 0; i < jset_line; i++)
            {
                J_index[0][i] = x.J_index[0][i];
                J_index[1][i] = x.J_index[1][i];
            }
            cblas_dcopy(jset_line, x.J_val, 1, J_val, 1);
        }
    }
    return *this;
}

// jset.txtの行数をカウントする
int Jset::count_lines()
{
    ifstream if_jset(jset_filename);

    if (if_jset.eof())
    {  //ファイルの終端まで読み込む場合
        if_jset.clear();
        if_jset.seekg(std::ios::beg);
    }
    else
    {
        if_jset.seekg(std::ios::beg);
    }

    int Jset_line = 0;
    std::string count_tmp;
    while (getline(if_jset, count_tmp))
    {
        Jset_line++;
    }
    if_jset.close();
    return Jset_line;
}

//ファイルが存在するかを調べる
bool Jset::file_check(std::string filename)
{
    ifstream ifj(filename);
    return !(ifj);
}

// Jsetの情報を出力する
void Jset::print() const
{
    cout << "\nfile name : " << jset_filename << std::endl;

    cout << "======================================\n";
    cout << setw(3) << "i" << setw(3) << "j" << setw(5) << "J[i][j]\n";
    cout << "--------------------------------------\n";
    for (int i = 0; i < jset_line; i++)
    {
        cout << setw(3) << J_index[0][i] << setw(3) << J_index[1][i] << setw(5)
             << J_val[i] << endl;
    }
    cout << "======================================\n";
}

//相互作用の情報を配列に格納する
void Jset::set()
{
    ifstream if_jset(jset_filename);

    if (if_jset.eof())
    {  //ファイルの終端まで読み込む場合
        if_jset.clear();
    }
    if_jset.seekg(std::ios::beg);

    string str_tmp;
    int J_ind_temp_i = 0, J_ind_temp_j = 0;
    double J_val_temp = 0;
    int line = 0;
    while (getline(if_jset, str_tmp))
    {
        stringstream ss;
        ss << str_tmp;
        ss >> J_ind_temp_i >> J_ind_temp_j >> J_val_temp;
        J_index[0][line] = J_ind_temp_i;
        J_index[1][line] = J_ind_temp_j;
        J_val[line] = J_val_temp;
        line++;
    }
}

//文字列表現を返却する
std::string Jset::to_string() const
{
    std::ostringstream s;

    s << "\nfile name : " << jset_filename << std::endl;

    s << "======================================\n";
    s << setw(3) << "i" << setw(3) << "j" << setw(10) << "J[i][j]\n";
    s << "--------------------------------------\n";
    for (int i = 0; i < jset_line; i++)
    {
        s << setw(3) << J_index[0][i] << setw(3) << J_index[1][i] << setw(7)
          << J_val[i] << endl;
    }
    s << "======================================\n";

    return s.str();
}
