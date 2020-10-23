#ifndef OPINION_OPT_NETWORK_HPP
#define OPINION_OPT_NETWORK_HPP

#include <string>
#include <vector>
#include <fstream>

using namespace std;

class Network{
  public:
    Network();
    Network(const string& file_to_read, const string& log_path, const bool& load_setup, const int& manual_seed, const unsigned short& n_threads);
    ~Network();
    void load_connectivity(const string& graph_file);
    void load_variables(const string& log_path);
    void load_vector(const string& file_to_write, vector<double>& vec);
    void load_matrix(const string& file_to_write, vector<vector<double>>& mat);
    void generate_rand_variables(const string& log_path, const int& manual_seed);
    void marginal_greedy(unsigned budget, const string& log_path);
    void random_batch(unsigned batch, unsigned budget, const string& log_path);
    void batch_gradient_greedy(unsigned batch, unsigned budget, const string& log_path);
    double get_sum(vector<double>& vec);
    void save_equilibrium(const string& file_to_write, const double& equilibrium = -1.0, bool overwrite = true);

    template <typename VecType>
    void record_vector(const string& file_to_write, const VecType& vec);
    void record_matrix(const string& file_to_write, const vector<vector<double>>& mat);

    void set_up_thread_points();
    void initialize_to_ones(vector<double>& vec);
    static void initialize_to_ones_thread(unsigned short thread_ID, vector<double>& vec);
    void compute_As();
    static void compute_As_thread(unsigned short thread_ID);
    static void update_z();
    static void compute_z_thread(unsigned short thread_ID);
    static void compute_Pz_thread(unsigned short thread_ID);
    static void update_b();
    static void compute_b_thread(unsigned short thread_ID);
    static void compute_b_minus_Ab_thread(unsigned short thread_ID);
    void update_L_and_J(vector<unsigned>& L,vector<unsigned>& J, vector<unsigned>& set);
    bool is_z_precise_enough(double& err);
    double compute_delta_lower_bound(unsigned& row, vector<double>& factors, double& err);
    double compute_delta_upper_bound(unsigned& row, vector<double>& factors, double& err);

    void log_time(ofstream& fout);

  private:
    unsigned num_nodes;
    unsigned num_edges;

    static unsigned short num_threads;
    static vector<unsigned> thread_points_for_edges;
    static vector<unsigned> thread_points_for_nodes;

    static vector<unsigned> num_neighbors;
    static vector<vector<unsigned>> neighbor_indexes;

    static vector<double> s;  // innate opinion vector
    static vector<double> alpha;  // resistance vector
    static vector<double> As;  // intermediate result As, where A = diag(alpha) 
    static vector<double> b_minus_Ab;
    vector<double> l;  // lower bound of alpha
    vector<double> u;  // upper bound of alpha
    static vector<double> z;  // opinion vector
    static vector<double> b;
    static vector<double> Pz; // opinion vector after one update
    static vector<vector<double>> P;
    static vector<vector<double>> P_trans;
};

#endif