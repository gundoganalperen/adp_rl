"""
    Approximate Dynamic Programming & Reinforcement Learning - WS 2018
    Programming Assignment
    Alperen Gundogan
    
    30.01.2019
    
    Command necessary to test the code.
    python main.py maze.txt
"""
from __future__ import division
import sys
import os
from maze_environment import Maze_env
from maze_solver import Maze_solver

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

MAX_ITER = 1000

if __name__ == '__main__':
    if len(sys.argv) < 2:
         print('Arguments: <input file> <output file>', len(sys.argv))
         sys.exit(1)
    script = sys.argv[0]
    input_file = sys.argv[1]

    maze_sol = Maze_solver()
    maze_sol.read_file(input_file)
    cost_functions = [1, 2]
    alpha = [0.99, 0.01, 0.3, 0.5, 0.7, 0.9]

    for g in cost_functions:

        maze_sol.maze_env.build_transition_probability_matrix(cost_function = g)
        for a in alpha:

            policy, v, it = maze_sol.policy_iteration(discount_factor = a, max_iteration=MAX_ITER)
            print("Policy Iteration: " + str(it) + " Cost function: " + str(g) + " Discount factor: " + str(a) )
            text = "PI"+ ",g" + str(g) + ",a=" + str(a)
            maze_sol.plot_error(maze_sol.gt_PI[g-1], maze_sol.values, text)
            maze_sol.visulaze_results(v, policy, text)


            policy, v, it = maze_sol.value_iteration(discount_factor = a, max_iteration=MAX_ITER)
            print("Value Iteration: " + str(it) + " Cost function: " + str(g) + " Discount factor: " + str(a))
            text_v = "VI"+ ",g" + str(g) + ",a=" + str(a)

            maze_sol.plot_error(maze_sol.gt_VI[g-1], maze_sol.values, text_v)
            maze_sol.visulaze_results(v, policy, text_v)

    plt.show()
