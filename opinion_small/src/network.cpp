#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <algorithm>
#include <sys/time.h>
#include <chrono>
#include <string>
#include <armadillo>

#include "network.hpp"

//using namespace std;
//using namespace arma;

Network::Network(){
}

Network::Network(const string& graph_file, const string& log_path, const bool& load_setup, const int& manual_seed){
    load_connectivity(graph_file);
    if(load_setup){
        load_variables(log_path);
    }else{
        generate_rand_variables(log_path, manual_seed);
    }
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

    // collect edge info from the graph file
    neighbor_indexes.resize(num_nodes);
    unsigned node_1 = 0;
    unsigned node_2 = 0;
    while (fin >> node_1 >> node_2) {
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


void Network::load_vector(const string& file_to_read, colvec& vector) {
    vector.zeros(num_nodes);
    ifstream fin(file_to_read);
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
        fin >> vector(row);
    }
    fin.close();
}

void Network::load_matrix(const string& file_to_write, mat& matrix) {
    matrix.zeros(num_nodes, num_nodes);
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
        for (unsigned column = 0; column < num_neighbors[row]; ++column) {
            fin >> matrix(row, neighbor_indexes[row][column]);
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

    s.zeros(num_nodes);
    for (unsigned row = 0; row < num_nodes; ++row) {
        s(row) = uniformDistribution(randNumGenerator);
    }

    l.zeros(num_nodes);
    u.zeros(num_nodes);
    for (unsigned row = 0; row < num_nodes; ++row) {
        if(uniformDistribution(randNumGenerator) < 0.99){
            l(row) = 0.001;
        }
        else{
            l(row) = 0.001 + 0.099 * uniformDistribution(randNumGenerator);
        }

        if(uniformDistribution(randNumGenerator) < 0.99){
            u(row) = 0.999;
        }
        else{
            u(row) = 0.9 + 0.099 * uniformDistribution(randNumGenerator);
        }

        if(l(row) >= u(row)){
            --row;
        }
    }

    P.zeros(num_nodes, num_nodes);
    for (unsigned row = 0; row < num_nodes; ++row) {
        for (unsigned column = 0; column < num_neighbors[row]; ++column) {
            P(row, neighbor_indexes[row][column]) = uniformDistribution(randNumGenerator);
        }
    }

    // normalize P
    for (unsigned row = 0; row < num_nodes; ++row) {
        double sum_of_row = 0;
        for (unsigned column = 0; column < num_nodes; ++column) {
            sum_of_row += P(row, column);
        }
        for (unsigned column = 0; column < num_nodes; ++column) {
            P(row, column) /= sum_of_row;
        }
    }

    alpha.zeros(num_nodes);
    for (unsigned row = 0; row < num_nodes; ++row) {
        alpha(row) = uniformDistribution(randNumGenerator);

        if(alpha(row) < l(row) || alpha(row) > u(row)){
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


void Network::marginal_greedy_inv(unsigned budget, const string& log_path){
    cout << " Start selecting alpha via marginal greedy inversion. " << endl;

    // Pre-load nodes to T if exists
    ifstream fin(log_path + "T_MGI.txt");
    vector<unsigned> T;
    unsigned node_index;
    while (fin >> node_index) {
        T.push_back(node_index);
        alpha(node_index) = u(node_index);
    }
    fin.close();
    cout << " Load " << T.size() << " nodes to T. " << endl;

    vector<unsigned> V;
    V.resize(num_nodes);
    for(unsigned row = 0; row < num_nodes; ++row) {
        V[row] = row;
    }

    double f = num_nodes;
    for(unsigned j = T.size(); j < budget; ++j){
        cout << " Selecting the " << j + 1 << "-th alpha. " << endl;

        unsigned selected_index = 0;
        colvec selected_alpha = alpha;
        colvec alpha_backup = alpha;
        for(unsigned i = 0; i < V.size(); ++i){
            alpha = alpha_backup;
            alpha(V[i]) = u(V[i]);

            vector<unsigned> L;
            vector<unsigned> J;
            bool is_z_precise = false;
            while (!is_z_precise){
                z = inv(diagmat(ones<colvec>(num_nodes)) - diagmat(ones<colvec>(num_nodes) - alpha) * P) * (alpha % s);

                for(unsigned index = 0; index < T.size(); ++index){
                    if(z(T[index]) <= s(T[index])){
                        if(alpha(T[index]) == u(T[index])){
                            L.push_back(T[index]);
                        }
                    }else{
                        if(alpha(T[index]) == l(T[index])){
                            J.push_back(T[index]);
                        }
                    }
                }

                if(z(V[i]) <= s(V[i])){
                    if(alpha(V[i]) == u(V[i])){
                        L.push_back(V[i]);
                    }
                }else{
                    if(alpha(V[i]) == l(V[i])){
                        J.push_back(V[i]);
                    }
                }

                if(!L.empty() || !J.empty()){
                    for(auto itr = L.begin(); itr != L.end(); ++itr){
                        alpha(*itr) = l(*itr);
                    }
                    L.clear();
                    for(auto itr = J.begin(); itr != J.end(); ++itr){
                        alpha(*itr) = u(*itr);
                    }
                    J.clear();
                }else{
                    is_z_precise = true;
                }
            }

            double sum_z = sum(z);
            if(f > sum_z){
                f = sum_z;
                selected_index = i;
                selected_alpha = alpha;
            }
        }

        T.push_back(V[selected_index]);
        save_equilibrium(log_path + "result_MGI.txt", V[selected_index], false);
        save_equilibrium(log_path + "result_MGI.txt", f, false);
        cout << " The " << T.size() << "-th selected node is " << V[selected_index] << endl;
        alpha = selected_alpha;
        V.erase(V.begin() + selected_index);
    }
    record_vector(log_path + "T_MGI.txt", T);
    cout << " Flipping finished! " << endl;
}

void Network::random_batch_inv(unsigned batch_size, unsigned budget, const string& log_path){
    cout << " Start selecting alpha via random batch inversion. " << endl;

    // Pre-load nodes to T if exists
    ifstream fin(log_path + "T_RBI.txt");
    vector<unsigned> T;
    unsigned node_index;
    while (fin >> node_index) {
        T.push_back(node_index);
        alpha(node_index) = u(node_index);
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
            save_equilibrium(log_path + "result_RBI.txt", node, false);
            cout << " The " << T.size() << "-th selected node is " << node << endl;
        }

        vector<unsigned> L;
        vector<unsigned> J;
        bool is_z_precise = false;
        while (!is_z_precise){
            z = inv(diagmat(ones<colvec>(num_nodes)) - diagmat(ones<colvec>(num_nodes) - alpha) * P) * (alpha % s);

            for(unsigned index = 0; index < T.size(); ++index){
                if(z(T[index]) <= s(T[index])){
                    if(alpha(T[index]) == u(T[index])){
                        L.push_back(T[index]);
                    }
                }else{
                    if(alpha(T[index]) == l(T[index])){
                        J.push_back(T[index]);
                    }
                }
            }

            if(!L.empty() || !J.empty()){
                for(auto itr = L.begin(); itr != L.end(); ++itr){
                    alpha(*itr) = l(*itr);
                }
                L.clear();
                for(auto itr = J.begin(); itr != J.end(); ++itr){
                    alpha(*itr) = u(*itr);
                }
                J.clear();
            }else{
                is_z_precise = true;
            }
        }
        save_equilibrium(log_path + "result_RBI.txt", -1.0, false);
    }
    record_vector(log_path + "T_RBI.txt", T);
    cout << " Flipping finished! " << endl;
}


void Network::batch_gradient_greedy_inv(unsigned batch_size, unsigned budget, const string& log_path){
    cout << " Start selecting alpha via " << batch_size <<"-batch gradient greedy inversion. " << endl;

    // Pre-load nodes to T if exists
    ifstream fin(log_path + "T_BGGI.txt");
    vector<unsigned> T;
    unsigned node_index;
    while (fin >> node_index) {
        T.push_back(node_index);
        alpha(node_index) = u(node_index);
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

    while(T.size() < budget){
        /* Uncomment this block and set batch_size to 100 or 1000 for running time test
        save_equilibrium(log_path + "time_BGGI_" + to_string(1) + "_" + to_string(num_nodes) + ".txt", -2, false);
        save_equilibrium(log_path + "time_BGGI_" + to_string(10) + "_" + to_string(num_nodes) + ".txt", -2, false);
        save_equilibrium(log_path + "time_BGGI_" + to_string(100) + "_" + to_string(num_nodes) + ".txt", -2, false);
        save_equilibrium(log_path + "time_BGGI_" + to_string(1000) + "_" + to_string(num_nodes) + ".txt", -2, false);
        //*/

        colvec b;
        vector<unsigned> L;
        vector<unsigned> J;
        bool is_z_precise = false;
        while(!is_z_precise){
            z = inv(diagmat(ones<colvec>(num_nodes)) - diagmat(ones<colvec>(num_nodes) - alpha) * P) * (alpha % s);
            b = inv(diagmat(ones<colvec>(num_nodes)) - trans(P) * diagmat(ones<colvec>(num_nodes) - alpha)) * ones<colvec>(num_nodes);

            for(unsigned index = 0; index < T.size(); ++index){
                if(z(T[index]) <= s(T[index])){
                    if(alpha(T[index]) == u(T[index])){
                        L.push_back(T[index]);
                    }
                }else{
                    if(alpha(T[index]) == l(T[index])){
                        J.push_back(T[index]);
                    }
                }
            }

            if(!L.empty() || !J.empty()){
                for(auto itr = L.begin(); itr != L.end(); ++itr){
                    alpha(*itr) = l(*itr);
                }
                L.clear();
                for(auto itr = J.begin(); itr != J.end(); ++itr){
                    alpha(*itr) = u(*itr);
                }
                J.clear();
            }else{
                is_z_precise = true;
            }
        }
        save_equilibrium(log_path + "result_BGGI_" + to_string(batch_size) + ".txt", -1.0, false);

        colvec factors = zeros<colvec>(num_nodes);
        for(unsigned i = 0; i < V.size(); ++i) {
            if(s(V[i]) > z(V[i])){
                factors(V[i]) = alpha(V[i]) - l(V[i]);
            }else{
                factors(V[i]) = alpha(V[i]) - u(V[i]);
            }
        }
        factors /= ones<colvec>(num_nodes) - alpha;
        colvec delta = factors % b % (s - z);

        unsigned num_of_new_nodes = 0;
        while(num_of_new_nodes < batch_size && T.size() < budget){
            unsigned selected_node = delta.index_max();
            delta(selected_node) = 0;
            T.push_back(selected_node);
            ++num_of_new_nodes;

            //* Comment this block and set batch_size to 100 or 1000 for running time test
            save_equilibrium(log_path + "result_BGGI_" + to_string(batch_size) + ".txt", selected_node, false);
            cout << " The " << T.size() << "-th selected node is " << selected_node << endl;
            //*/

            if(s(selected_node) > z(selected_node)){
                alpha(selected_node) = l(selected_node);
            }else{
                alpha(selected_node) = u(selected_node);
            }
            V.erase(find(V.begin(), V.end(), selected_node));

            /* Uncomment this block and set batch_size to 100 or 1000 for running time test
            if(num_of_new_nodes == 1) save_equilibrium(log_path + "time_BGGI_" + to_string(1) + "_" + to_string(num_nodes) + ".txt", -2, false);
            if(num_of_new_nodes == 10) save_equilibrium(log_path + "time_BGGI_" + to_string(10) + "_" + to_string(num_nodes) + ".txt", -2, false);
            if(num_of_new_nodes == 100) save_equilibrium(log_path + "time_BGGI_" + to_string(100) + "_" + to_string(num_nodes) + ".txt", -2, false);
            if(num_of_new_nodes == 1000) save_equilibrium(log_path + "time_BGGI_" + to_string(1000) + "_" + to_string(num_nodes) + ".txt", -2, false);
            //*/
        }
        factors.clear();

        if(T.size() == budget){
            is_z_precise = false;
            while (!is_z_precise){
                z = inv(diagmat(ones<colvec>(num_nodes)) - diagmat(ones<colvec>(num_nodes) - alpha) * P) * (alpha % s);

                for(unsigned index = 0; index < T.size(); ++index){
                    if(z(T[index]) <= s(T[index])){
                        if(alpha(T[index]) == u(T[index])){
                            L.push_back(T[index]);
                        }
                    }else{
                        if(alpha(T[index]) == l(T[index])){
                            J.push_back(T[index]);
                        }
                    }
                }

                if(!L.empty() || !J.empty()){
                    for(auto itr = L.begin(); itr != L.end(); ++itr){
                        alpha(*itr) = l(*itr);
                    }
                    L.clear();
                    for(auto itr = J.begin(); itr != J.end(); ++itr){
                        alpha[*itr] = u[*itr];
                    }
                    J.clear();
                }else{
                    is_z_precise = true;
                }
            }
            save_equilibrium(log_path + "result_BGGI_" + to_string(batch_size) + ".txt", -1.0, false);
        }
    }
    record_vector(log_path + "T_BGGI.txt", T);
    cout << " Flipping finished! " << endl;
}

void Network::save_equilibrium(const string& file_to_write, const double& equilibrium, bool overwrite){
    double sum_z = 0.0;
    if(equilibrium == -1.0){
        sum_z = sum(z);
    }else{
        sum_z = equilibrium;
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
    fout << sum_z << endl;
    fout.close();
}

template <typename VecType>
void Network::record_vector(const string& file_to_write, const VecType& vector){
    ofstream fout(file_to_write);
    fout.setf(ios::fixed, ios::floatfield);
    fout.precision(10);
    for(unsigned i = 0; i < num_nodes; ++i){
        fout << vector[i] << endl;
    }
    fout.close();
}

void Network::record_matrix(const string& file_to_write, const mat& matrix){
    ofstream fout(file_to_write);
    fout.setf(ios::fixed, ios::floatfield);
    fout.precision(10);
    for (unsigned row = 0; row < num_nodes; ++row) {
        for (unsigned column = 0; column < num_neighbors[row]; ++column) {
            fout << matrix(row, neighbor_indexes[row][column]) << endl;
        }
    }
    fout.close();
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

