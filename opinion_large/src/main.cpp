
#include <string.h>
#include <getopt.h>
#include <iostream>
#include <filesystem>
#include <sys/time.h>

#include "network.hpp"

using namespace std;
using namespace std::filesystem;

int main(int argc, char *argv[]) {

    string graph_file = "";
    string load_setup_path = "";
    int algorithm = 'g';
    unsigned batch_size = 1;
    unsigned budget = 1;
    bool load_setup = false;
    int manual_seed = -1;
    unsigned short num_threads = 1;

    static struct option long_options[] =
    {
        {"algorithm",       required_argument, NULL, 'a'},
        {"budget",          required_argument, NULL, 'b'},
        {"graph_file",      required_argument, NULL, 'g'},
        {"load_setup_path", required_argument, NULL, 'l'},
        {"manual_seed",     required_argument, NULL, 'm'},
        {"batch_size",      required_argument, NULL, 's'},
        {"num_threads",     required_argument, NULL, 't'},
    };

    int opt;
    const char *opt_string = "a:b:g:l:m:s:t:";
    while((opt = getopt_long(argc, argv, opt_string, long_options, NULL)) != -1) {
        switch(opt) {
            case 'a':
                if(strcmp(optarg, "bgg") == 0){
                    algorithm = 'g';
                    cout << "algorithm: batch gradient greedy " << endl;
                }else if(strcmp(optarg, "mg") == 0){
                    algorithm = 'm';
                    cout << "algorithm: marginal greedy " << endl;
                }else if(strcmp(optarg, "rn") == 0){
                    algorithm = 'r';
                    cout << "algorithm: random node selection " << endl;
                }else{
                    cout << "Error: wrong argument a " << endl;
                    return 1;
                }
                break;
            case 'b':
                budget = atoi(optarg);
                cout << "budget: " << budget << endl;
                break;
            case 'g':
                graph_file = optarg;
                cout << "graph_file: " << graph_file << endl;
                break;
            case 'l':
                load_setup_path = optarg;
                cout << "load_setup_path: " << load_setup_path << endl;
                load_setup = true;
                break;
            case 'm':
                manual_seed = atoi(optarg);
                if(manual_seed < 0){
                    cout << "Error: wrong argument m/manual_seed. Try another positive integer." << endl;
                    return 1;
                }
                cout << "manual_seed: " << manual_seed << endl;
                break;
            case 's':
                batch_size = atoi(optarg);
                cout << "batch_size: " << batch_size << endl;
                break;
            case 't':
                num_threads = atoi(optarg);
                cout << "num_threads: " << num_threads << endl;
                break;
            default:
                cout << "Error: wrong argument " << char(optopt) << endl;
                return 1;
        }
    }

    string log_path;
    if(load_setup){
        log_path = load_setup_path + "/";
    }else{
        char buffer[26];
        struct tm* tm_info;
        struct timeval tv;
        gettimeofday(&tv, NULL);
        tm_info = localtime(&tv.tv_sec);
        strftime(buffer, 26, "%Y_%m_%d_%H_%M_%S", tm_info);
        log_path = buffer;
        log_path = "./results/" + log_path;

        path pathObj(graph_file);
        if (pathObj.has_stem()) {
            log_path += "_" + pathObj.stem().string() + "/";
        }else{
            log_path += "/";
        }

        create_directories(log_path);
    }
    cout << "log_path: " << log_path << endl;

    Network network(graph_file, log_path, load_setup, manual_seed, num_threads);

    switch(algorithm) {
        case 'g':
//            cout << "algorithm: batch gradient greedy " << endl;
            network.batch_gradient_greedy(batch_size, budget, log_path);
            break;
        case 'm':
//            cout << "algorithm: marginal greedy " << endl;
            network.marginal_greedy(budget, log_path);
            break;
        case 'r':
//            cout << "algorithm: random node selection " << endl;
            network.random_batch(batch_size, budget, log_path);
            break;
        default:
            cout << "Error: wrong algorithm " << endl;
            return 1;
    }

    return 0;
}
