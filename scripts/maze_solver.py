"""
    Approximate Dynamic Programming & Reinforcement Learning - WS 2018
    Programming Assignment
    Alperen Gundogan
    
    30.01.2019
    
    Command necessary to test the code.
    python main.py maze.txt
"""
#from __future__ import division
from maze_environment import Maze_env

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Determine whether goal state skipped or not.
# This is implemented just for some tests.
SKIP_GOAL_STATE = 0
GRUND_TRUTH = 0.99

class Maze_solver(object):

    def __init__(self):
        self.maze_env = Maze_env()
        self.policy = []
        self.values = []
        # initialize as zero.
        # holds the grund truth for both g1 and g2 respectively.
        self.gt_PI= []
        self.gt_VI = []
        self.algorithm_name = ""


    def read_file(self, path):
        """
        Read the maze.
        :param path:
        :return:
        """
        maze = []
        with open(path) as f:
            for line in f.read().splitlines():
                line = line.replace(" ", "")
                if not line.startswith("#"):
                    maze.append(list(line))
        self.maze_env.maze = maze
        self.maze_env.maze = np.asarray(self.maze_env.maze)
        self.maze_env.set_environment()

    def policy_evaluation(self, discount_factor, max_iteration, theta=1e-25):
        """
        Runs the policy evaluation algorithm.
        :param discount_factor:
        :param max_iteration:
        :param theta:
        :return: returns the converged value function and total number of iterations.
        """
        V = np.zeros(self.maze_env.nS)
        it_eval = 0
        for i in range(max_iteration):
            delta = 0
            it_eval += 1
            # For each state, perform a "full backup"
            for s in range(self.maze_env.nS):
                v = 0
                if s in self.maze_env.walls:
                    continue
                # Look at the possible next actions
                for a, action_prob in enumerate(self.policy[s]):
                    # For each action, look at the possible next states...
                    for prob, next_state, reward, done in self.maze_env.P[s][a]:
                        # Calculate the expected value.
                        v += action_prob * prob * (reward + discount_factor * V[next_state])
                # How much our value function changed (across any states)
                delta = max(delta, np.abs(v - V[s]))
                V[s] = v
            # Stop evaluating once our value function change is below a threshold
            if delta < theta:
                break
        return np.array(V), it_eval

    def policy_iteration(self, discount_factor, max_iteration):
        """
        Runs the policy iteration algorithm. We first create random policy and then
        runs policy evaluation and policy improvement algorithms, respectively.
        :param discount_factor:
        :param max_iteration:
        :return: Returns the optimal policy and cost function with the total number of iterations.
        """

        self.policy = self.create_random_policy()
        self.values = []
        it = 0
        while True:
            # Evaluation of current policy
            V, iter = self.policy_evaluation(discount_factor, max_iteration)
            # this will be set to false if we make any changes.
            self.values.append(V[self.maze_env.start])
            optimal_policy = True
            it += 1
            for s in range(self.maze_env.nS):
                # the walls also part of the state but we skip it since there can be no actions inside the wall.
                if s in self.maze_env.walls:
                    continue

                chosen_a = np.argmax(self.policy[s])

                A = self.values_of_actions(s, V, discount_factor)

                # Choose the best action which minimize the value function
                best_a = np.argmin(A)

                # Greedily update the policy
                if chosen_a != best_a:
                    optimal_policy = False
                self.policy[s] = np.eye(self.maze_env.nA)[best_a]

            if optimal_policy or it == max_iteration:
                if discount_factor == GRUND_TRUTH:
                    self.gt_PI.append(V[self.maze_env.start])
                return self.policy, V, it

    def values_of_actions(self, state, V, discount_factor):
        """
        For the given value function and the state and discount factor, returns
        the values of each available action on that state.
        :param state:
        :param V:       values of state, array.
        :param discount_factor:
        :return:        values of each available action on that state
        """
        # Find the values of each action by looking successor states.
        A = np.zeros(self.maze_env.nA)
        av_ac = self.maze_env.available_actions(state)
        for a in range(self.maze_env.nA):
            if a in av_ac:
                for prob, next_state, reward, done in self.maze_env.P[state][a]:
                    A[a] += prob * (reward + discount_factor * V[next_state])
            else:
                A[a] = np.inf

        return A

    def value_iteration(self, discount_factor, max_iteration, theta=1e-25):
        """
        Runs the value iteration algorithm.
        :param discount_factor:
        :param max_iteration:
        :param theta:
        :return:    Returns the optimal policy, value function and total number of iteration for algorithm.
        """
        V = np.zeros(self.maze_env.nS)
        it = 0
        while it != max_iteration:
            # Condition for stop
            delta = 0
            # increase the iteration
            it += 1
            # Update each state
            for s in range(self.maze_env.nS):
                # the walls also part of the state but we skip it since there can be no actions inside the wall.
                if s in self.maze_env.walls:
                    continue
                # Find the values of each action by looking successor states.
                A = self.values_of_actions(s, V, discount_factor)
                best_action_value = np.min(A)
                # Calculate delta across all states seen so far
                delta = max(delta, np.abs(best_action_value - V[s]))
                # Update the value function. Ref: Sutton book eq. 4.10.
                V[s] = best_action_value
            self.values.append(V[self.maze_env.t])
            # Check if we can stop
            if delta < theta:
                break

        # Create "the" policy based on the optimal value function.
        policy = np.zeros([self.maze_env.nS, self.maze_env.nA])

        for s in range(self.maze_env.nS):
            # the walls also part of the state but we skip it since there can be no actions inside the wall.
            if s in self.maze_env.walls:
                continue
            # Find the best action for this state using the corresponding values of each action.
            A = self.values_of_actions(s, V, discount_factor)
            best_action = np.argmin(A)
            # Always take the best action
            policy[s, best_action] = 1.0

        if discount_factor == GRUND_TRUTH:
            self.gt_VI.append(V[self.maze_env.start])
        return policy, V, it

    def create_random_policy(self):
        """
        This function creates uniform random policy.
        :return:
        """
        #self.policy = np.zeros([self.maze_env.nS, self.maze_env.nA])
        policy = np.zeros([self.maze_env.nS, self.maze_env.nA])
        it = np.nditer(self.maze_env.grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            # Skip the transition probabilities for the wall.
            if self.maze_env.maze[y][x] == '1':
                it.iternext()
                continue
            # determine the available actions for the given state.
            actions = self.maze_env.available_actions(s)
            for a in actions:
                policy[s][a] = 1.0/len(actions)

            it.iternext()
        return policy

    def show_maze(self):
        """
        This function call shows the maze, if it is needed.
        :return:
        """
        plt.grid('on')
        nrows, ncols = self.maze_env.maze.shape
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows+1, 1))
        ax.set_yticks(np.arange(0.5, ncols+1, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        canvas = np.ones((nrows, ncols))

        for var in range(len(self.maze_env.goal_loc[0])):
            row = self.maze_env.goal_loc[0][var]
            col = self.maze_env.goal_loc[1][var]
            canvas[row,col] = 0.5
        for var in range(len(self.maze_env.start_loc[0])):
            row = self.maze_env.start_loc[0][var]
            col = self.maze_env.start_loc[1][var]
            canvas[row,col] = 0.3
        for var in range(len(self.maze_env.traps_loc[0])):
            row = self.maze_env.traps_loc[0][var]
            col = self.maze_env.traps_loc[1][var]
            canvas[row,col] = 0.7
        for var in range(len(self.maze_env.walls_loc[0])):
            row = self.maze_env.walls_loc[0][var]
            col = self.maze_env.walls_loc[1][var]
            canvas[row,col] = 0.0

        canvas = np.array(canvas, dtype=float)

        img = plt.imshow(canvas, interpolation='none', cmap='gray')

        return img

    def plot_value_function(self, text):
        """
        Plots the value function at start position with the number of iterations.
        :param text:
        :return:
        """
        plt.clf()
        plt.plot(self.values)
        plt.ylabel('Values')
        plt.xlabel('Iterations')
        plt.title(text)
        plt.show()

    def plot_error(self, gt, values, title):
        """
        Plots the squared distance error.
        :param gt:  Ground truth
        :param values: Values of start position w.r.t the number of iterations.
        :param title:
        :return:
        """
        #plt.clf()
        fig = plt.figure()
        errors = []
        #gt = gt *  np.ones(len(values))
        #dist = np.linalg.norm(gt - values)
        for i in range(len(values)):
            errors.append(np.sqrt((gt-values[i])**2))

        iter = range(1, len(values)+1)

        #
        plt.plot(iter, errors)
        plt.ylabel('Squared Distance')
        plt.xlabel('Iterations')
        plt.title(title)
        plt.tight_layout()
        title = title + "_error"
        directory = "plots/"
        #plt.savefig(title+"."+"png", format="png", dpi = 1200)
        #plt.savefig(directory+title+"."+"pdf", format="pdf", dpi = 1200)
        #plt.show()

    def visulaze_results(self, V, p, title, save = False, format = "png",):
        """
        This function creates heatmap and quiver plot for the given value function and policy.

        :param V: "The" (optimal) value function.
        :param p: "The" (optimal) policy.
        :param title: Title of the plot.
        :param save:
        :param format:
        :return:
        """
        #plt.clf()
        fig, ax = plt.subplots()
        nrows = self.maze_env.max_y
        ncols = self.maze_env.max_x
        V = np.reshape(V, (nrows, ncols))
        p_shaped = np.reshape(np.argmax(p, axis=1), self.maze_env.shape)
        for var in range(len(self.maze_env.walls_loc[0])):
            row = self.maze_env.walls_loc[0][var]
            col = self.maze_env.walls_loc[1][var]
            V[row][col] = np.nan
            p_shaped[row][col] = -1
        # masked array to hold walls
        masked_array = np.ma.array (V, mask=np.isnan(V))
        current_cmap = cm.get_cmap()
        current_cmap.set_bad('black')
        im = ax.imshow(masked_array, cmap=current_cmap)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(title)


        y_pos = self.maze_env.ava_states[0]
        x_pos = self.maze_env.ava_states[1]
        x_direct, y_direct = self.helper_quiver_plot(p_shaped)
        ax.quiver(x_pos,y_pos,x_direct,y_direct, scale=20)

        plt.tight_layout()

        #directory = "plots/"
        #plt.savefig(title+"."+"png", format="png", dpi = 1200)
        #plt.savefig(directory+title+"."+"pdf", format="pdf", dpi = 1200)


    def helper_quiver_plot(self, p_shaped):
        """
        This function helps to plot quiver.
        :param p_shaped: policy array with the shape of the maze.
        :return:
        """
        x_direct = []
        y_direct = []

        for j in range(self.maze_env.max_y):
            for i in range(self.maze_env.max_x):
                # skip if it is a wall
                if p_shaped[j][i] == -1:
                    continue
                # add the up.
                if p_shaped[j][i] == 0:
                    x_direct.append(0)
                    y_direct.append(1)
                # Right.
                elif p_shaped[j][i] == 1:
                    x_direct.append(1)
                    y_direct.append(0)
                # Down.
                elif p_shaped[j][i] == 2:
                    x_direct.append(0)
                    y_direct.append(-1)
                # Right.
                elif p_shaped[j][i] == 3:
                    x_direct.append(-1)
                    y_direct.append(0)
                # Idle
                elif p_shaped[j][i] == 4:
                    x_direct.append(0)
                    y_direct.append(0)

        return x_direct, y_direct



