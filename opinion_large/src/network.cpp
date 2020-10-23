#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <algorithm>
#include <sys/time.h>
#include <chrono>
#include <string>

#include "network.hpp"

//using namespace std;
//using namespace arma;

vector<unsigned> Network::thread_points_for_edges;
vector<unsigned> Network::thread_points_for_nodes;
vector<unsigned> Network::num_neighbors;
vector<double> Network::s;
vector<double> Network::alpha;
vector<double> Network::As;
vector<double> Network::b_minus_Ab;
vector<vector<unsigned>> Network::neighbor_indexes;
vector<double> Network::z;
vector<double> Network::b;
vector<double> Network::Pz;
vector<vector<double>> Network::P;
vector<vector<double>> Network::P_trans;
unsigned short Network::num_threads;

Network::Network(){
}

Network::Network(const string& graph_file, const string& log_path, const bool& load_setup, const int& manual_seed, const unsigned short& n_threads){
    num_threads = n_threads;
    load_connectivity(graph_file);
    if(load_setup){
        load_variables(log_path);
    }else{
        generate_rand_variables(log_path, manual_seed);
    }
    set_up_thread_points();
}

Network::~Network(){
}

void Network::load_connectivity(const string& graph_file) {
    ifstream fin(graph_file);
    try{
        if (!fin) {
            throw "Can't read file !";
        }
    }
    catch (const char* exp) {
        cout << "Exception: " << exp << endl;
        cout << "End program!" << endl;
        abort();
    }

    cout << " File opened, start collecting graph info. " << endl;
    fin >> num_nodes;

    // collect edge info from the data file
    neighbor_indexes.resize(num_nodes);
    num_edges = 0;
    unsigned node_1 = 0;
    unsigned node_2 = 0;
    while (fin >> node_1 >> node_2) {
        ++num_edges;
        if ((node_1 >= num_nodes) || (node_2 >= num_nodes)) {
            cout << " Vertex ID exceeds # of nodes! " << endl;
            cout << " Vertex 1 = " << node_1 << endl;
            cout << " Vertex 2 = " << node_2 << endl;
            cout << " # of nodes = " << num_nodes << endl;
        }
        neighbor_indexes[node_1].push_back(node_2);
        neighbor_indexes[node_2].push_back(node_1);
    }
    fin.close();

    num_neighbors.resize(num_nodes);
    for (unsigned row = 0; row < num_nodes; ++row) {
        num_neighbors[row] = neighbor_indexes[row].size();
        neighbor_indexes[row].resize(num_neighbors[row]);
    }

    cout << " Graph info collection finished! " << endl;
    cout << " number of nodes: " << num_nodes << endl;
    cout << " number of threads: " << num_threads << endl;
}


void Network::load_variables(const string& log_path) {
    cout << " Loading s, l, u, alpha and P. " << endl;

    load_vector(log_path + "s", s);
    load_vector(log_path + "l", l);
    load_vector(log_path + "u", u);
    load_vector(log_path + "alpha", alpha);
    load_matrix(log_path + "P", P);

    cout << " Loading finished! " << endl;
}


void Network::load_vector(const string& file_to_write, vector<double>& vec) {
    vec.resize(num_nodes);
    ifstream fin(file_to_write);
    try{
        if (!fin) {
            throw "Can't read file !";
        }
    }
    catch (const char* exp) {
        cout << "Exception: " << exp << endl;
        cout << "End program!" << endl;
        abort();
    }

    for (unsigned row = 0; row < num_nodes; ++row) {
        fin >> vec[row];
    }
    fin.close();
}

void Network::load_matrix(const string& file_to_write, vector<vector<double>>& mat) {
    mat.resize(num_nodes);
    ifstream fin(file_to_write);
    try{
        if (!fin) {
            throw "Can't read file !";
        }
    }
    catch (const char* exp) {
        cout << "Exception: " << exp << endl;
        cout << "End program!" << endl;
        abort();
    }

    for (unsigned row = 0; row < num_nodes; ++row) {
        mat[row].resize(num_neighbors[row]);
        for (unsigned column = 0; column < num_neighbors[row]; ++column) {
            fin >> mat[row][column];
        }
    }
    fin.close();
}

void Network::generate_rand_variables(const string& log_path, const int& manual_seed) {
    cout << " Initializing s, l, u, alpha and P randomly. " << endl;

    // parameters for random number generation
    unsigned seed;
    if (manual_seed < 0){
        seed = chrono::system_clock::now().time_since_epoch().count();
    }else{
        seed = manual_seed;
        cout << " Using manual_seed: " << manual_seed << endl;
    }

    default_random_engine randNumGenerator(seed);
    uniform_real_distribution<double> uniformDistribution(0.0, 1.0);
    //cout << " test rand num: " << uniformDistribution(randNumGenerator) << endl;

    s.resize(num_nodes);
    for (unsigned row = 0; row < num_nodes; ++row) {
        s[row] = uniformDistribution(randNumGenerator);
    }

    l.resize(num_nodes);
    u.resize(num_nodes);
    for (unsigned row = 0; row < num_nodes; ++row) {
        if(uniformDistribution(randNumGenerator) < 0.99){
            l[row] = 0.001;
        }
        else{
            l[row] = 0.001 + 0.099 * uniformDistribution(randNumGenerator);
        }

        if(uniformDistribution(randNumGenerator) < 0.99){
            u[row] = 0.999;
        }
        else{
            u[row] = 0.9 + 0.099 * uniformDistribution(randNumGenerator);
        }

        if(l[row] >= u[row]){
            --row;
        }
    }

    P.resize(num_nodes);
    for (unsigned row = 0; row < num_nodes; ++row) {
        P[row].resize(num_neighbors[row]);
        for (unsigned column = 0; column < num_neighbors[row]; ++column) {
            P[row][column] = uniformDistribution(randNumGenerator);
        }
    }

    // normalize P
    for (unsigned row = 0; row < num_nodes; ++row) {
        double sum_of_row = 0;
        for (unsigned column = 0; column < num_neighbors[row]; ++column) {
            sum_of_row += P[row][column];
        }
        for (unsigned column = 0; column < num_neighbors[row]; ++column) {
            P[row][column] /= sum_of_row;
        }
    }

    alpha.resize(num_nodes);
    for (unsigned row = 0; row < num_nodes; ++row) {
        alpha[row] = uniformDistribution(randNumGenerator);

        if(alpha[row] < l[row] || alpha[row] > u[row]){
            --row;
        }
    }

    record_vector(log_path + "s", s);
    record_vector(log_path + "l", l);
    record_vector(log_path + "u", u);
    record_vector(log_path + "alpha", alpha);
    record_matrix(log_path + "P", P);

    cout << " Initialization finished! " << endl;
}

void Network::marginal_greedy(unsigned budget, const string& log_path){
    cout << " Start selecting alpha via marginal greedy. " << endl;

    // Pre-load nodes to T if exists
    ifstream fin(log_path + "T_MG.txt");
    vector<unsigned> T;
    unsigned node_index;
    while (fin >> node_index) {
        T.push_back(node_index);
        alpha[node_index] = u[node_index];
    }
    fin.close();
    cout << " Load " << T.size() << " nodes to T. " << endl;

    vector<unsigned> V;
    V.resize(num_nodes);
    for(unsigned row = 0; row < num_nodes; ++row) {
        V[row] = row;
    }
    for(unsigned i = 0; i < T.size(); ++i){
        V.erase(V.begin() + distance(V.cbegin(), find(V.cbegin(), V.cend(), T[i])));
    }

    double f = num_nodes;
    for(unsigned j = T.size(); j < budget; ++j){
        cout << " Selecting " << j + 1 << "-th alpha. " << endl;

        unsigned selected_index = 0;
        vector<double> selected_alpha = alpha;
        vector<double> alpha_backup = alpha;
        for(unsigned i = 0; i < V.size(); ++i){
            alpha = alpha_backup;
            alpha[V[i]] = u[V[i]];

            double min_alpha = *min_element(alpha.begin(), alpha.end());
            double err = 1.0 / min_alpha;
            vector<unsigned> L;
            vector<unsigned> J;

            initialize_to_ones(ref(z));
            compute_As();
            Pz.resize(num_nodes);

            while (!is_z_precise_enough(ref(err))){
                update_z();
                err *= 1.0 - min_alpha;
                update_L_and_J(L, J, T);
                if(z[V[i]] <= s[V[i]]){
                    if(alpha[V[i]] == u[V[i]]){
                        L.push_back(V[i]);
                    }
                }
                else{
                    if(alpha[V[i]] == l[V[i]]){
                        J.push_back(V[i]);
                    }
                }

                if(!L.empty() || !J.empty()){
                    for(auto itr = L.begin(); itr != L.end(); ++itr){
                        alpha[*itr] = l[*itr];
                        As[*itr] = alpha[*itr] * s[*itr];
                        if(l[*itr] < min_alpha){
                            min_alpha = l[*itr];
                        }
                    }
                    L.clear();
                    for(auto itr = J.begin(); itr != J.end(); ++itr){
                        alpha[*itr] = u[*itr];
                        As[*itr] = alpha[*itr] * s[*itr];
                        if(l[*itr] == min_alpha){
                            min_alpha = *min_element(alpha.begin(), alpha.end());
                        }
                    }
                    J.clear();
                    err = 1.0 / min_alpha;
                }
            }
            Pz.clear();
            As.clear();

            double sum_z = get_sum(z);
            if(f > sum_z){
                f = sum_z;
                selected_index = i;
                selected_alpha = alpha;
            }
        }

        alpha_backup.clear();
        T.push_back(V[selected_index]);
        save_equilibrium(log_path + "result_MG.txt", V[selected_index], false);
        save_equilibrium(log_path + "result_MG.txt", f, false);
        cout << " The " << T.size() << "-th selected node is " << V[selected_index] << endl;
        alpha = selected_alpha;
        V.erase(V.begin() + selected_index);
        selected_alpha.clear();
    }
    record_vector(log_path + "T_MG.txt", T);
    cout << " Flipping finished! " << endl;
}

void Network::random_batch(unsigned batch_size, unsigned budget, const string& log_path){
    cout << " Start selecting alpha via random batch. " << endl;

    // Pre-load nodes to T if exists
    ifstream fin(log_path + "T_RB.txt");
    vector<unsigned> T;
    unsigned node_index;
    while (fin >> node_index) {
        T.push_back(node_index);
        alpha[node_index] = u[node_index];
    }
    fin.close();
    cout << " Load " << T.size() << " nodes to T. " << endl;

    vector<unsigned> V;
    V.resize(num_nodes);
    for(unsigned row = 0; row < num_nodes; ++row) {
        V[row] = row;
    }
    for(unsigned i = 0; i < T.size(); ++i){
        V.erase(V.begin() + distance(V.cbegin(), find(V.cbegin(), V.cend(), T[i])));
    }

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    while(T.size() < budget){
        shuffle(V.begin(), V.end(), default_random_engine(seed));
        for(unsigned i = 0; i < batch_size && T.size() < budget; ++i){
            unsigned node = V.back();
            V.pop_back();
            alpha[node] = u[node];
            T.push_back(node);
            save_equilibrium(log_path + "result_RB.txt", node, false);
            cout << " The " << T.size() << "-th selected node is " << node << endl;
        }

        double min_alpha = *min_element(alpha.begin(), alpha.end());
        double err = 1.0 / min_alpha;
        vector<unsigned> L;
        vector<unsigned> J;

        initialize_to_ones(ref(z));
        compute_As();
        Pz.resize(num_nodes);

        while (!is_z_precise_enough(ref(err))){
            update_z();
            err *= 1.0 - min_alpha;

            update_L_and_J(L, J, T);

            if(!L.empty() || !J.empty()){
                for(auto itr = L.begin(); itr != L.end(); ++itr){
                    alpha[*itr] = l[*itr];
                    As[*itr] = alpha[*itr] * s[*itr];
                    if(l[*itr] < min_alpha){
                        min_alpha = l[*itr];
                    }
                }
                L.clear();
                for(auto itr = J.begin(); itr != J.end(); ++itr){
                    alpha[*itr] = u[*itr];
                    As[*itr] = alpha[*itr] * s[*itr];
                    if(l[*itr] == min_alpha){
                        min_alpha = *min_element(alpha.begin(), alpha.end());
                    }
                }
                J.clear();
                err = 1.0 / min_alpha;
            }
        }
        Pz.clear();
        As.clear();
        save_equilibrium(log_path + "result_RB.txt", -1.0, false);
    }
    record_vector(log_path + "T_RB.txt", T);
    cout << " Flipping finished! " << endl;
}

void Network::batch_gradient_greedy(unsigned batch_size, unsigned budget, const string& log_path){
    cout << " Start selecting alpha via " << batch_size <<"-batch gradient greedy. " << endl;

    // Pre-load nodes to T if exists
    ifstream fin(log_path + "T_BGG.txt");
    vector<unsigned> T;
    unsigned node_index;
    while (fin >> node_index) {
        T.push_back(node_index);
        alpha[node_index] = u[node_index];
    }
    fin.close();
    cout << " Load " << T.size() << " nodes to T. " << endl;

    vector<unsigned> V;
    V.resize(num_nodes);
    for(unsigned row = 0; row < num_nodes; ++row) {
        V[row] = row;
    }
    for(unsigned i = 0; i < T.size(); ++i){
        V.erase(V.begin() + distance(V.cbegin(), find(V.cbegin(), V.cend(), T[i])));
    }

    P_trans.resize(num_nodes);
    for (unsigned row = 0; row < num_nodes; ++row) {
        P_trans[row].resize(num_neighbors[row]);
        for (unsigned column = 0; column < num_neighbors[row]; ++column) {
            unsigned transpose_row = neighbor_indexes[row][column];
            unsigned transpose_column = distance(neighbor_indexes[transpose_row].cbegin(),
                                        find(neighbor_indexes[transpose_row].cbegin(),
                                            neighbor_indexes[transpose_row].cend(),
                                            row));
            P_trans[row][column] = P[transpose_row][transpose_column];
        }
    }
    Pz.resize(num_nodes);
    compute_As();
    double min_alpha = *min_element(alpha.begin(), alpha.end());
    initialize_to_ones(ref(z));

    while(T.size() < budget){
        /* Uncomment this block and set batch_size and budget to 100 or 1000 for running time test
        save_equilibrium(log_path + "time_BGG_" + to_string(1) + "_" + to_string(num_nodes) + ".txt", -2, false);
        save_equilibrium(log_path + "time_BGG_" + to_string(10) + "_" + to_string(num_nodes) + ".txt", -2, false);
        save_equilibrium(log_path + "time_BGG_" + to_string(100) + "_" + to_string(num_nodes) + ".txt", -2, false);
        save_equilibrium(log_path + "time_BGG_" + to_string(1000) + "_" + to_string(num_nodes) + ".txt", -2, false);
        //*/

        initialize_to_ones(ref(b));
        vector<unsigned> L;
        vector<unsigned> J;
        double err = 1.0 / min_alpha;
        b_minus_Ab.resize(num_nodes);

        while(!is_z_precise_enough(ref(err))){
            thread update_z_thread = thread(update_z);
            thread update_b_thread = thread(update_b);
            update_z_thread.join();
            update_b_thread.join();
            err *= 1.0 - min_alpha;

            update_L_and_J(L, J, T);

            if(!L.empty() || !J.empty()){
                for(auto itr = L.begin(); itr != L.end(); ++itr){
                    alpha[*itr] = l[*itr];
                    As[*itr] = alpha[*itr] * s[*itr];
                    if(l[*itr] < min_alpha){
                        min_alpha = l[*itr];
                    }
                }
                L.clear();
                for(auto itr = J.begin(); itr != J.end(); ++itr){
                    alpha[*itr] = u[*itr];
                    As[*itr] = alpha[*itr] * s[*itr];
                    if(l[*itr] == min_alpha){
                        min_alpha = *min_element(alpha.begin(), alpha.end());
                    }
                }
                J.clear();
                err = 1.0 / min_alpha;
            }
        }
        save_equilibrium(log_path + "result_BGG_" + to_string(batch_size) + ".txt", -1.0, false);

        vector<double> factors;
        factors.resize(num_nodes);
        for(unsigned i = 0; i < V.size(); ++i) {
            if(s[V[i]] > z[V[i]]){
                factors[V[i]] = alpha[V[i]] - l[V[i]];
            }else{
                factors[V[i]] =  u[V[i]] - alpha[V[i]];
            }
            factors[V[i]] /= (1.0 - alpha[V[i]]);
        }

        unsigned num_of_new_nodes = 0;
        while(num_of_new_nodes < batch_size && T.size() < budget){
            double max_delta_upper_bound = -1;
            double max_delta_lower_bound = compute_delta_lower_bound(V[0], factors, err);
            unsigned selected_index = 0;

            for(unsigned i = 1; i < V.size(); ++i){
                double delta_lower_bound = compute_delta_lower_bound(V[i], factors, err);
                if(max_delta_lower_bound < delta_lower_bound){
                    max_delta_lower_bound = delta_lower_bound;
                    double delta_upper_bound = compute_delta_upper_bound(V[selected_index], factors, err);
                    if(max_delta_upper_bound < delta_upper_bound){
                        max_delta_upper_bound = delta_upper_bound;
                    }
                    selected_index = i;
                }else{
                    double delta_upper_bound = compute_delta_upper_bound(V[i], factors, err);
                    if(max_delta_upper_bound < delta_upper_bound){
                        max_delta_upper_bound = delta_upper_bound;
                    }
                }
            }
            if(max_delta_lower_bound >= max_delta_upper_bound){
                T.push_back(V[selected_index]);
                ++num_of_new_nodes;

                //* Comment this block and set batch_size and budget to 100 or 1000 for running time test
                save_equilibrium(log_path + "result_BGG_" + to_string(batch_size) + ".txt", V[selected_index], false);
                cout << " The " << T.size() << "-th selected node is " << V[selected_index] << endl;
                //*/

                if(s[V[selected_index]] > z[V[selected_index]]){
                    alpha[V[selected_index]] = l[V[selected_index]];
                    As[V[selected_index]] = alpha[V[selected_index]] * s[V[selected_index]];
                    if(l[V[selected_index]] < min_alpha){
                        min_alpha = l[V[selected_index]];
                    }
                }else{
                    alpha[V[selected_index]] = u[V[selected_index]];
                    As[V[selected_index]] = alpha[V[selected_index]] * s[V[selected_index]];
                    if(l[V[selected_index]] == min_alpha){
                        min_alpha = *min_element(alpha.begin(), alpha.end());
                    }
                }
                V.erase(V.begin() + selected_index);

                /* Uncomment this block and set batch_size and budget to 100 or 1000 for running time test
                if(num_of_new_nodes == 1) save_equilibrium(log_path + "time_BGG_" + to_string(1) + "_" + to_string(num_nodes) + ".txt", -2, false);
                if(num_of_new_nodes == 10) save_equilibrium(log_path + "time_BGG_" + to_string(10) + "_" + to_string(num_nodes) + ".txt", -2, false);
                if(num_of_new_nodes == 100) save_equilibrium(log_path + "time_BGG_" + to_string(100) + "_" + to_string(num_nodes) + ".txt", -2, false);
                if(num_of_new_nodes == 1000) save_equilibrium(log_path + "time_BGG_" + to_string(1000) + "_" + to_string(num_nodes) + ".txt", -2, false);
                //*/
            }else{
                thread update_z_thread = thread(update_z);
                thread update_b_thread = thread(update_b);
                update_z_thread.join();
                update_b_thread.join();
                err *= 1.0 - min_alpha;
            }
        }
        factors.clear();
        b_minus_Ab.clear();


        if(T.size() == budget){
            err = 1.0 / min_alpha;
            while (!is_z_precise_enough(ref(err))){
                update_z();
                err *= 1.0 - min_alpha;

                update_L_and_J(L, J, T);

                if(!L.empty() || !J.empty()){
                    for(auto itr = L.begin(); itr != L.end(); ++itr){
                        alpha[*itr] = l[*itr];
                        As[*itr] = alpha[*itr] * s[*itr];
                        if(l[*itr] < min_alpha){
                            min_alpha = l[*itr];
                        }
                    }
                    L.clear();
                    for(auto itr = J.begin(); itr != J.end(); ++itr){
                        alpha[*itr] = u[*itr];
                        As[*itr] = alpha[*itr] * s[*itr];
                        if(l[*itr] == min_alpha){
                            min_alpha = *min_element(alpha.begin(), alpha.end());
                        }
                    }
                    J.clear();
                    err = 1.0 / min_alpha;
                }
            }
            save_equilibrium(log_path + "result_BGG_" + to_string(batch_size) + ".txt", -1.0, false);
        }
    }
    Pz.clear();

    record_vector(log_path + "T_BGG.txt", T);
    cout << " Flipping finished! " << endl;
}

double Network::get_sum(vector<double>& vec){
    double sum = 0;
    for (auto itr = vec.begin(); itr != vec.end(); ++itr) {
        sum += *itr;
    }
    return sum;
}

void Network::save_equilibrium(const string& file_to_write, const double& equilibrium, bool overwrite){
    double sum = 0.0;
    if(equilibrium == -1.0){
        sum = get_sum(z);
    }else{
        sum = equilibrium;
    }

    ofstream fout;
    if(overwrite){
        fout.open(file_to_write);
    }else{
        fout.open(file_to_write, ios::app);
    }
    log_time(fout);
    fout.setf(ios::fixed, ios::floatfield);
    fout.precision(16);
    fout << sum << endl;
    fout.close();
}

template <typename VecType>
void Network::record_vector(const string& file_to_write, const VecType& vec){
    ofstream fout(file_to_write);
    fout.setf(ios::fixed, ios::floatfield);
    fout.precision(10);
    for(unsigned i = 0; i < vec.size(); ++i){
        fout << vec[i] << endl;
    }
    fout.close();
}

void Network::record_matrix(const string& file_to_write, const vector<vector<double>>& mat){
    ofstream fout(file_to_write);
    fout.setf(ios::fixed, ios::floatfield);
    fout.precision(10);
    for(unsigned i = 0; i < mat.size(); ++i){
        for(unsigned j = 0; j < mat[i].size(); ++j){
            fout << mat[i][j] << endl;
        }
    }
    fout.close();
}


void Network::set_up_thread_points() {
    unsigned interval = num_edges * 2 / num_threads;
    unsigned sum_of_size = 0;

    thread_points_for_edges.push_back(0);
    for (unsigned row = 0; row < num_nodes; ++row) {
        sum_of_size += num_neighbors[row];
        if (num_neighbors[row] > interval) {
            cout << "interval: " << interval << endl;
            cout << "num_neighbors: " << num_neighbors[row] << endl;
            cout << "index: " << row << endl;
        }
        if (sum_of_size > interval) {
            thread_points_for_edges.push_back(row);
            sum_of_size = 0;
        }
    }
    thread_points_for_edges.push_back(num_nodes);
    if (thread_points_for_edges.size() != (num_threads + 1)) {
        cout << "Edge thread point setting failed! " << endl;
    }

    interval = num_nodes / num_threads;
    for (unsigned short thread_ID = 0; thread_ID < num_threads; ++thread_ID) {
        thread_points_for_nodes.push_back(interval * thread_ID);
    }
    thread_points_for_nodes.push_back(num_nodes);
    if (thread_points_for_nodes.size() != (num_threads + 1)) {
        cout << "Node thread point setting failed! " << endl;
    }
}

void Network::initialize_to_ones(vector<double>& vec){
    vec.resize(num_nodes);
    vector<thread> multi_thread;
    for (unsigned short thread_ID = 0; thread_ID < num_threads; ++thread_ID) {
        multi_thread.push_back(thread(initialize_to_ones_thread, thread_ID, ref(vec)));
    }
    for (unsigned short thread_ID = 0; thread_ID < num_threads; ++thread_ID) {
        multi_thread[thread_ID].join();
    }
    multi_thread.clear();
}

void Network::initialize_to_ones_thread(unsigned short thread_ID, vector<double>& vec) {
    for (unsigned row = thread_points_for_nodes[thread_ID]; row < thread_points_for_nodes[thread_ID + 1]; ++row) {
        vec[row] = 1.0;
    }
}

void Network::compute_As(){
    As.resize(num_nodes);
    vector<thread> multi_thread;
    for (unsigned short thread_ID = 0; thread_ID < num_threads; ++thread_ID) {
        multi_thread.push_back(thread(compute_As_thread, thread_ID));
    }
    for (unsigned short thread_ID = 0; thread_ID < num_threads; ++thread_ID) {
        multi_thread[thread_ID].join();
    }
    multi_thread.clear();
}

void Network::compute_As_thread(unsigned short thread_ID){
    for (unsigned row = thread_points_for_nodes[thread_ID]; row < thread_points_for_nodes[thread_ID + 1]; ++row) {
        As[row] = alpha[row] * s[row];
    }
}

void Network::update_z(){
    vector<thread> multi_thread;
    for (unsigned short thread_ID = 0; thread_ID < num_threads; ++thread_ID) {
        multi_thread.push_back(thread(compute_Pz_thread, thread_ID));
    }
    for (unsigned short thread_ID = 0; thread_ID < num_threads; ++thread_ID) {
        multi_thread[thread_ID].join();
    }
    multi_thread.clear();

    for (unsigned short thread_ID = 0; thread_ID < num_threads; ++thread_ID) {
        multi_thread.push_back(thread(compute_z_thread, thread_ID));
    }
    for (unsigned short thread_ID = 0; thread_ID < num_threads; ++thread_ID) {
        multi_thread[thread_ID].join();
    }
    multi_thread.clear();
}

void Network::compute_z_thread(unsigned short thread_ID) {
    for (unsigned row = thread_points_for_nodes[thread_ID]; row < thread_points_for_nodes[thread_ID + 1]; ++row) {
        z[row] = As[row] + Pz[row] * (1.0 - alpha[row]);
    }
}

void Network::compute_Pz_thread(unsigned short thread_ID) {
    for (unsigned row = thread_points_for_edges[thread_ID]; row < thread_points_for_edges[thread_ID + 1]; ++row) {
        Pz[row] = 0.0;
        for (unsigned column = 0; column < num_neighbors[row]; ++column) {
            Pz[row] += P[row][column] * z[neighbor_indexes[row][column]];
        }
    }
}

void Network::update_b(){
    vector<thread> multi_thread;
    for (unsigned short thread_ID = 0; thread_ID < num_threads; ++thread_ID) {
        multi_thread.push_back(thread(compute_b_minus_Ab_thread, thread_ID));
    }
    for (unsigned short thread_ID = 0; thread_ID < num_threads; ++thread_ID) {
        multi_thread[thread_ID].join();
    }
    multi_thread.clear();

    for (unsigned short thread_ID = 0; thread_ID < num_threads; ++thread_ID) {
        multi_thread.push_back(thread(compute_b_thread, thread_ID));
    }
    for (unsigned short thread_ID = 0; thread_ID < num_threads; ++thread_ID) {
        multi_thread[thread_ID].join();
    }
    multi_thread.clear();
}

void Network::compute_b_thread(unsigned short thread_ID) {
    for (unsigned row = thread_points_for_edges[thread_ID]; row < thread_points_for_edges[thread_ID + 1]; ++row) {
        b[row] = 0.0;
        for(unsigned column = 0; column < num_neighbors[row]; ++column){
            b[row] += P_trans[row][column] * b_minus_Ab[neighbor_indexes[row][column]];
        }
        b[row] += 1.0;
    }
}

void Network::compute_b_minus_Ab_thread(unsigned short thread_ID) {
    for (unsigned row = thread_points_for_nodes[thread_ID]; row < thread_points_for_nodes[thread_ID + 1]; ++row) {
        b_minus_Ab[row] = (1.0 - alpha[row]) * b[row];
    }
}

void Network::update_L_and_J(vector<unsigned>& L,vector<unsigned>& J, vector<unsigned>& set) {
    for(auto itr = set.begin(); itr != set.end(); ++itr){
        if(z[*itr] <= s[*itr]){
            if(alpha[*itr] == u[*itr]){
                L.push_back(*itr);
            }
        }
        else{
            if(alpha[*itr] == l[*itr]){
                J.push_back(*itr);
            }
        }
    }
}

bool Network::is_z_precise_enough(double& err) {
    for (unsigned row = 0; row < num_nodes; ++row) {
        if (abs(z[row] - s[row]) <= err) {
            return false;
        }
    }
    return true;
}

double Network::compute_delta_lower_bound(unsigned& row, vector<double>& factors, double& err){
    return factors[row] * (b[row] - err * num_nodes) * (abs(s[row] - z[row]) - err);
}

double Network::compute_delta_upper_bound(unsigned& row, vector<double>& factors, double& err){
    return factors[row] * (b[row] + err * num_nodes) * (abs(s[row] - z[row]) + err);
}

void Network::log_time(ofstream& fout) {
    char buffer[26];
    int millisec;
    struct tm* tm_info;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    millisec = lrint(tv.tv_usec / 1000.0); // Round to nearest millisec
    if (millisec >= 1000){ // Allow for rounding up to nearest second
        millisec -= 1000;
        tv.tv_sec++;
    }

    tm_info = localtime(&tv.tv_sec);

    strftime(buffer, 26, "%Y:%m:%d %H:%M:%S", tm_info);
    fout << buffer << "." ;
    if(millisec < 100) fout << "0";
    if(millisec < 10) fout << "0";
    fout << millisec << endl;
}
