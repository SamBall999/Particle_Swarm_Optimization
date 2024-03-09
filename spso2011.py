# SPSSO2011

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pickle
from benchmark_functions import spherical, ackley, michalewicz, katsuura, shubert, r_s_ackley, r_s_michalewicz, r_s_katsuura, r_s_shubert




class Particle:

    """
    Individual in the swarm containing own position, velocity, personal best, neighbourhood and neighbourhood best.

    """

    def __init__(self, position, dimension):

        """
        Initializes particle with the given position and dimension.

        Arguments:
        - Initial position of particle 
        - No. of dimensions 

        """

        self.position = position # current position of particle
        self.velocity = np.zeros(dimension) # initialize velocity to zero #
        self.p_best = position # initialize personal best to initial position of particle
        self.neighbours = None # neighbourhood of particles for communication
        self.n_best = None # best position found by neighbourhood

    


class SPSO2011():

    """
    Contains methods to perform standard particle swarm optimization (SPSO2011) for a given optimization function.

    """

    def __init__(self, lower_bound, upper_bound, function, rotate):

        """
        Initializes SPSO2011 instance with the given search space bounds, optimization function and rotation information.

        Arguments:
        - Lower boundary of the search space
        - Upper boundary of the search space
        - Objective function to be minimized
        - Number representing whether the function is shifted and rotated (0 - no transformation, 1 - shifted and rotated)

        """

        self.M = None
        self.o = None
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.function = function
        self.rotate = rotate




    def get_transformations(self):

        """
        Sets rotation matrix and shift to be applied to objective function.


        Returns:
        - Rotation matrix M
        - Shift vector o

        """

        # read in possible rotation matrices
        with open(os.path.join(os.path.dirname(__file__), 'data.pkl'), 'rb') as _pkl_file:
            _pkl = pickle.load(_pkl_file)


        M_all = _pkl['M_D20']
        M = M_all[0] # choose first rotation matrix
        self.M = M

    
        o_array = [np.random.uniform(self.lower_bound, self.upper_bound, 20)]
        o = o_array[0]
        #print("o")
        #print(o)
        self.o = o

        return M, o




    def obj_function(self, x):


        """
        Calculates fitness of the current particle position for the objective function being minimized.

        Arguments:
        - Current position of the particle

        Returns:
        - Fitness of the particle.

        """
    
    
        # check if function is shifted and rotated
        if (self.rotate == 1):
            fitness = self.function(x, self.M, self.o)
        else:
            fitness = self.function(x)

        return fitness 




    def H(self, centre, radius, n):

        """
        Returns a uniformly sampled random position from a hypersphere centred at gi(t) with a radius of norm((gi - xi)).

        Arguments:
        - Centre of gravity
        - Radius of hypersphere 
        - Number of dimensions n

        Returns:
        - Uniformly sampled random position from the hypersphere

        """


        # normal dist - to ensure normalization does not destroy the uniformity of the sampling - to avoid centre bias
        # generate d scalars from normal dist and construct the vector re from d scalars
        re = [np.random.normal(0, 1, n)] # n is the no. of dimensions - make sure this is correct

        # normalize re
        re_normalized = re/np.linalg.norm(re) # is this the correct norm???

        # generate scalar S from U(0, ra)
        s = np.random.uniform(0, radius)

        # scale and shift re to correct position in search space
        re_final = s*(re_normalized) + centre

        return re_final




    
    def velocity_update(self, w, c_1, r_1, c_2, r_2, x_i, p_i, n_i, v_current):

        """
        Updates the velocity of the particle based on hypersphere distribution.

        Arguments:
        - Inertia weight w
        - Cognitive coefficient c1
        - Random vector r1
        - Social coefficient c2
        - Random vector r2
        - Current position of particle
        - Personal best of particle
        - Neighbourhood best 
        - Current velocity of particle

        Returns:
        - New particle velocity

        """

        # g - collective best
        # p - personal best (particle i)
        # h(g_i, norm) - hypersphere distr. centred at g_i with radius being the distance between gi and xi


        a_i =  x_i + np.multiply(c_1*r_1, p_i - x_i)  # cognitive info
        b_i = x_i + np.multiply(c_2*r_2, n_i - x_i) # social info
        g_i = (x_i + a_i + b_i)/3

        norm = np.linalg.norm(g_i - x_i)
        v_next = w*v_current + self.H(g_i, norm, len(x_i)) - x_i # g - centre of gravity

        return v_next[0]



  
    def find_neighbourhood(self, particle, population):

        """
        Selects k neighbours for the given particle using the adaptive random topology.

        Arguments:
        - Current particle
        - Swarm of all particles

        Returns:
        - Neighbourhood of the given particle.

        """

        # select 3 random particles from swarm (inc. particle itself)

        k = 3
        indices = np.random.randint(0, len(population)-1, k)
        neighbourhood = [population[index] for index in indices]
        particle.neighbours = neighbourhood # set variable in particle

        return neighbourhood




    def get_neighbourhood_best(self, particle):

        """
        Finds the best solution in the neighbourhood of the particle.

        Arguments:
        - Current particle

        Returns:
        - The best particle in the neighbourhood (n_best)

        """

        neighbours = particle.neighbours
        fitnesses = []
        for i in range(len(neighbours)):

            fitnesses.append(self.obj_function(neighbours[i].position))

        index = np.argmin(fitnesses)
        particle.n_best = neighbours[index].position 

        return neighbours[index]


    def visualize_search(self, g_bests, vs, xis):

        """
        Plots visualisations of the change in global best, and velocity and position change of a particle, over all iterations.

        Arguments:
        - Global bests over the course of the search.
        - Velocity of particle 0 during the search.
        - Position of particle 0 during the search.

        """

        plt.plot(g_bests)
        plt.ylabel("Fitness")
        plt.xlabel("Iterations")
        plt.show()

        plt.plot(vs)
        plt.ylabel("Velocity")
        plt.xlabel("Iterations")
        plt.show()

        plt.plot(xis)
        plt.ylabel("Particle")
        plt.xlabel("Iterations")
        plt.show()


    def run_spso2011(self):

        """
        Performs standard particle swarm optimization 2011

        Returns:
        - Best overall solution found during the search.

        """


        # tune for each function
        w = 0.729844
        #w = 0.5
        c_1 = 1.49618
        #c_1 = 1.9
        #c_2 = 1.9
        c_2 = 1.49618

        print("w: {}".format(w))
        print("c1=c2: {}".format(c_1))

        # init population
        dimension = 20
        population_size = 20
        # sampled from a predefined hypercube - dependent on search space 
        initial_positions = [np.random.uniform(self.lower_bound, self.upper_bound, dimension) for individual in range(population_size)] # initialize nx-dimensional population of given population size ns

        # create particles
        population = []
        for i in range(population_size):
            new_particle = Particle(initial_positions[i], dimension)
            population.append(new_particle)
    
        # initialize global best
        g_current = Particle(initial_positions[0], dimension) # first member of population set to global bets to start
        
        # store information for visualization
        global_bests = []
        global_bests.append(self.obj_function(g_current.position))
        velocities = []
        x_is = []

        # set neighbourhoods
        neighbourhoods = [self.find_neighbourhood(particle_i, population) for particle_i in population] # assign neighbours to each particle
        n_bests = [self.get_neighbourhood_best(particle_i) for particle_i in population]
        global_improved = 0 # used to reset neighbourhoods if global best does not improve

        # stopping condition
        t = 0
        max_iterations = 1000 #1000

        # assuming minimization
        while(t < max_iterations): 


            # update personal bests
            for i in range(len(population)):
                particle = population[i]
                x_i = particle.position
                p_i = particle.p_best
                if ((all(x > self.lower_bound for x in x_i)) and (all(x < self.upper_bound for x in x_i))): # boundary handling - do not update the personal and neighborhood best positions if they violate the search boundary. 
                    if(self.obj_function(x_i) < self.obj_function(p_i)):
                        particle.p_best = x_i # population[i] current index in population
        

                # inform particle itself
                n_i = particle.n_best
                if(self.obj_function(p_i) < self.obj_function(n_i)):
                    particle.n_best = p_i 

                # inform neighbours 
                neighbourhood = particle.neighbours
                for j in range(len(neighbourhood)): # k = 3 -> iterate over the neighbours
                    neighbour_particle = neighbourhood[j]
                    neighbour_best = neighbour_particle.n_best
                
                    # update their best as well
                    if(self.obj_function(p_i) < self.obj_function(neighbour_best)):
                        neighbour_particle.n_best = p_i
                        # update global best
                        if(self.obj_function(neighbour_particle.n_best) < self.obj_function(g_current.position)):
                            g_current.position = neighbour_particle.n_best #update position
                            global_improved = 1
                            #print("Global improved")

    
        
            # update velocity and position
            for i in range(len(population)):
                
                particle = population[i]
                x_i = particle.position
                p_i = particle.p_best
                n_i = particle.n_best

                # initialize random vectors r_1 and r_2 
                r_1 = np.random.uniform(0, 1, 20) # must be a 20-dim vector
                r_2 = np.random.uniform(0, 1, 20)

        
                particle.velocity = self.velocity_update(w, c_1, r_1, c_2, r_2, x_i, p_i, n_i, particle.velocity)
                particle.position = x_i + particle.velocity

            # if unsuccessful iteration occurs, neighbourhoods reconstructed - if global best did not improve 
            if (global_improved==0):
                neighbourhoods = [self.find_neighbourhood(particle_i, population) for particle_i in population] 
                #print("\n New neighbourhoods")
                #n_bests = [get_neighbourhood_best(neighbourhood, function) for neighbourhood in neighbourhoods] #do these get reset as well??
                #n_bests = [self.get_neighbourhood_best(particle_i) for particle_i in population]
            else:
                global_improved = 0 # reset

        
            global_bests.append(self.obj_function(g_current.position))
            velocities.append(population[0].velocity)
            x_is.append(population[0].position)
            t+=1
    

        #self.visualize_search(global_bests, velocities, x_is)
        return g_current

            




def main():

    """
    Runs the SPSO2011 algorithm for the given objective function.

    Arguments:
    - Objective function to be minimised

    """

    print("\SPSO2011")
    print("--------Benchmarking---------")

    if(len(sys.argv) > 1):
        print(str(sys.argv[1]))
        obj_func = str(sys.argv[1])
    else:
        print("\nIncorrect number of required parameters")
        print("Using default parameters: ")
        print("- spherical objective function")
        obj_func = "-sph"

    
    if(obj_func == "-sph"):
        lower_bound, upper_bound = -5.12, 5.12
        function = spherical
        rotate = 0
        print("Objective function: Spherical")
    
    elif(obj_func == "-a"):
        lower_bound, upper_bound = -32.768, 32.768
        function = ackley
        rotate = 0
        print("Objective function: Ackley")

    elif(obj_func == "-m"):
        lower_bound, upper_bound = 0, np.pi
        function = michalewicz
        rotate = 0
        print("Objective function: Michalewicz")
    
    elif(obj_func == "-k"):
        lower_bound, upper_bound = 0, 100
        function = katsuura
        rotate = 0
        print("Objective function: Katsuura")
    
    elif(obj_func == "-sh"):
        lower_bound, upper_bound = -10, 10
        function = shubert
        rotate = 0
        print("Objective function: Shubert")
    
    elif(obj_func == "-ar"):
        lower_bound, upper_bound = -32.768, 32.768
        function = r_s_ackley
        rotate = 1
        print("Objective function: Shifted and Rotated Ackley")
    
    elif(obj_func == "-mr"):
        lower_bound, upper_bound = 0, np.pi
        function = r_s_michalewicz
        rotate = 1
        print("Objective function: Shifted and Rotated Michalewicz")
    
    elif(obj_func == "-kr"):
        lower_bound, upper_bound = 0, 100
        function = r_s_katsuura
        rotate = 1
        print("Objective function: Shifted and Rotated Katsuura")
    
    elif(obj_func == "-shr"):
        lower_bound, upper_bound = -10, 10
        function = r_s_shubert
        rotate = 1
        print("Objective function: Shifted and Rotated Shubert")
    
    else:
        # default is spherical
        lower_bound, upper_bound = -5.12, 5.12
        function = spherical
        rotate = 0
        print("Objective function: Spherical")

    



    all_best_fitnesses = []

    for i in range(30):

        spso2011 = SPSO2011(lower_bound, upper_bound, function, rotate)
        spso2011.get_transformations()
        best = spso2011.run_spso2011()
        #print("Best Solution: {}".format(best.position))
        best_fitness = spso2011.obj_function(best.position)
        print("Fitness: {}\n".format(best_fitness))
        #print(best_fitness)
        all_best_fitnesses.append(best_fitness)
    

    print(all_best_fitnesses)
    print("Mean: {}".format(np.mean(all_best_fitnesses)))



if __name__ == "__main__":
    main()