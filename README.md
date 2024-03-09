# Particle Swarm Optimization
Comparing the performance of inertia weight PSO and SPSO2011 on classic optimization functions.

# Comparing the Performance of IPSO and SPSO2011 

The project is made up of the following files:
- ipso.py: Contains functions to run the inertia weight PSO algorithm
- spso2011.py: Contains functions to run the IPSO algorithm
- benchmark_functions.py: Contains definitions of the objective functions
- performance_data.py: Plots distributions of performance data and performs Mann-Whitney U test


## Inertia Weight PSO

To run the inertia weight PSO algorithm, the command line format is as follows:

*python3 ipso.py [objective_function_flag] [topology]*

The objective function flags are defined as follows:
- -sph : Spherical function
- -a : Ackley function
- -m : Michalewicz function
- -k : Katsuura function
- -sh : Shubert function 

The rotated variants are then denoted with an r suffix:

- -ar : Ackley function (Rotated and Shifted)
- -mr : Michalewicz function (Rotated and Shifted)
- -kr : Katsuura function (Rotated and Shifted)
- -shr : Shubert function (Rotated and Shifted)


The choice of topology is as follows:

- 0 - global (GBest) topology
- 1 - local ring topology (LBest)
- 2- local stochastic star topology (not used for experiments)


For example, to run the ackley function with a global best topology:

*python3 ipso.py -sph 0* 

The algorithm will run 30 times by default.


## SPSO2011

To run the SPSO2011 algorithm, the command line format is as follows:

*python3 ipso.py [objective_function_flag]*

where the objective function flags are defined in the same way as above. 
No topology needs to be specified as SPSO2011 has an adpative random topology by default.

*python3 spso2011.py -sph* 


For both functions, if no command line arguments are given, the default parameters will be used.
The default parameters are the spherical objective function and global topology in the case of the IPSO algorithm.

The algorithm will run 30 times by default.


## Rotations and Shifts

In order for the rotations and shifts to be applied to the objective functions, the file "data.pkl" must be in the same directory since it contains the rotation matrices applied to the functions.
