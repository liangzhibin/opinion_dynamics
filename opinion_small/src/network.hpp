#ifndef OPINION_OPT_NETWORK_HPP
#define OPINION_OPT_NETWORK_HPP

#include <string>
#include <vector>
#include <fstream>
#include <armadillo>

using namespace std;
using namespace arma;

class Network{
  public:
    Network();
    Network(const string& file_to_read, const string& log_path, const bool& load_setup, const int& manual_seed);
    ~Network();
    void load_connectivity(const string& file_to_read);
    void load_variables(const string& file_name);
    void load_vector(const string& file_to_write, colvec& vector);
    void load_matrix(const string& file_to_write, mat& matrix);
    void generate_rand_variables(const string& log_path, const int& manual_seed);
    void marginal_greedy_inv(unsigned budget, const string& log_path);
    void random_batch_inv(unsigned batch_size, unsigned budget, const string& log_path);
    void batch_gradient_greedy_inv(unsigned batch_size, unsigned budget, const string& log_path);
    void save_equilibrium(const string& file_to_write, const double& equilibrium = -1.0, bool overwrite = true);
    template <typename VecType>
    void record_vector(const string& file_to_write, const VecType& vector);
    void record_matrix(const string& file_to_write, const mat& matrix);
    void log_time(ofstream& fout);

  private:
    unsigned num_nodes;

    vector<unsigned> num_neighbors;
    vector<vector<unsigned>> neighbor_indexes;

    colvec s;      // innate opinion vector
    colvec alpha;  // resistance vector
    colvec l;      // lower bound of alpha
    colvec u;      // upper bound of alpha
    mat P;

    colvec z;
};

#endif