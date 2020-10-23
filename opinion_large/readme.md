# This program approximates the equilibrium opinion vector using the random walk recurrence

## Test environment
Ubuntu 20.04

Required packages:
cmake make gcc-9 g++-9

## How to run experiments
1. Compile the source:

       make
    or

       make main

2. Run the generated main with required parameters. Here are some examples.
    
    To use batch gradient greedy:
    
       ./main -a bgg -b 217 -g ./datasets/graph_217 -m 1 -s 1 -t 2
    To use marginal greedy:

       ./main -a mg -b 217 -g ./datasets/graph_217 -m 1 -t 2
    To use random node selection strategy:

       ./main -a rn -b 217 -g ./datasets/graph_217 -m 1 -s 1 -t 2
       
3. Example to load setup (s, l, u, alpha and P):

       ./main -a bgg -b 217 -g ./datasets/graph_217 -l ./results/2020_10_13_11_13_23_graph_217/ -s 1  -t 2

4. Parameters:

    -a (string) - algorithm: bgg, mg or rn. Default: bgg
    
    -b (unsigned) - budget. Default: 1

    -g (string) - graph file containing connectivity.  Default: empty string.

    -m (int) - manual seed, must be a positive integer. Default: time seed.

    -s (unsigned) - batch size. Default: 1.

    -l (string) - path to load setup (s, l, u, alpha and P). Default: empty string.
    
    -t (unsigned short) - number of multi-threads. Default: 1.
    
5. Running time results for marginal greed are obtained in result_MG.txt.
To collect Running time results for batch gradient greedy,
please comment or uncomment the related lines shown in network.cpp.

6. To generate alpha based on power law distribution,
please move power_law_generator.py to the folder containing files u and l for lower and upper bounds info.
Then run power_law_generator.py to generate the file alpha.

7. To explain the graph file format, take ./datasets/graph_217 as example:

       217  <-- total number of nodes
       0 1  <-- two node indexes as an edge (node index starting from 0)
       0 2
       0 3
       0 4
       0 5
       0 6
       0 7
       0 8
       0 16
       ...