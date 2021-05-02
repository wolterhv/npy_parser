#include "npy_parser.hpp"

int main(int argc, char *argv[]) 
{
    Eigen::MatrixXd lut;
    Eigen::VectorXd row;
    std::ifstream lut_fobj; 
    lut_fobj.open(argv[1], std::ifstream::in | std::ifstream::binary);
    // populate_lookup_table(&lut_fobj, &lut);
    // std::cout << lut << std::endl;
    // populate_lookup_row(&lut_fobj, &row, atoi(argv[2]));
    // std::cout << row  << std::endl;
    Metadata metadata;
    std::vector<double> vec;
    populate_vector<double>(&lut_fobj, &metadata, &vec);
    for (auto a : vec) {
        std::cout << a << ", ";
    }
    std::cout << std::endl;
    lut_fobj.close();
    return 0;
}
