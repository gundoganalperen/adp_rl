"""
    Approximate Dynamic Programming & Reinforcement Learning - WS 2018
    Programming Assignment
    Alperen Gundogan
    
    30.01.2019
    
    Command necessary to test the code.
    python main.py maze.txt
"""
import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
IDLE = 4

# Simulation parameters.
# define the cost as minus and maximize the value function.
COST_TRAP = 50
COST_ENERGY = 1
GOAL_REACHED = -1
# This is the probability of reaching adjacent states when the action is executed.
PRO = 0.1
SKIP_GOAL_STATE = 0

class Maze_env(object):

    def __init__(self):
        # Matrix of lists to store the maze.
        self.maze = []
        # Stores the maze with iteration numbers e.g. [1 2 3 ..\n 11 12 ... 79].
        self.grid = []
        # Stores the maze as a vector form.
        self.maze_vec = []
        # Stores the dimensions of maze
        self.shape = []
        # number of states
        self.nS = 0
        # number of actions = {UP, RIGHT, DOWN, LEFT, IDLE}.
        self.nA = 5
        # Maximum x direction of maze.
        self.max_y = 0
        # Maximum y direction of maze.
        self.max_x = 0
        # transition probability matrix.
        self.P = {}
        # location of start.
        self.start = 0
        # location of goal in the maze.
        self.goal = 0
        # Stores all the wall locations in the tuple.
        self.walls = ()
        # Stores all the trap locations in the tuple.
        self.traps = ()
        # Stores the goal location using coordinates e.g. (x, y) = (0, 1)
        self.goal_loc = ()
        # Stores the start location using coordinates e.g. (x, y) = (0, 1)
        self.start_loc = ()
        # Stores all the wall locations using coordinates e.g. (x, y) = (0, 1)
        self.walls_loc = ()
        # Stores all the trap locations using coordinates e.g. (x, y) = (0, 1)
        self.traps_loc = ()
        # Stores the available states
        self.ava_states = ()


    def set_environment(self):
        """
        Set the variables of environment.
        :return:
        """
        self.shape = self.maze.shape
        self.nS = np.prod(self.shape)

        self.max_y = self.shape[0]
        self.max_x = self.shape[1]
        self.grid = np.arange(self.nS).reshape(self.shape)

        self.ava_states = np.where(self.maze != '1')

        itemindex_g = np.where(self.maze=='G')
        self.goal_loc = itemindex_g
        self.goal = self.grid[itemindex_g[0][0]][itemindex_g[1][0]]

        itemindex_s = np.where(self.maze=='S')
        self.start_loc = itemindex_s
        self.start = self.grid[itemindex_s[0][0]][itemindex_s[1][0]]

        itemindex_1 = np.where(self.maze=='1')
        self.walls_loc = itemindex_1
        wall_list = []
        for var in range(len(self.walls_loc[0])):
            wall_list.append(self.grid[itemindex_1[0][var]][itemindex_1[1][var]])
        self.walls = tuple(wall_list)

        itemindex_t = np.where(self.maze=='T')
        self.traps_loc = itemindex_t
        trap_list = []
        for var in range(len(self.traps_loc[0])):
            trap_list.append(self.grid[itemindex_t[0][var]][itemindex_t[1][var]])
        self.traps = tuple(trap_list)

        self.maze_vec = self.maze.flatten()


    def build_transition_probability_matrix(self, cost_function):
        """
        Set the transition probabilities for the given maze and const function
        :param cost_function:   set the transition probability matrix using the given cost function.

        :return:
        """
        it = np.nditer(self.grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            # Skip the transition probabilities for the wall
            if self.maze[y][x] == '1':
                it.iternext()
                continue

            # P[s][a] = (prob, next_state, reward, is_done)
            self.P[s] = {a : [] for a in range(self.nA)}

            is_goal_reached = lambda s: s == self.goal

            # We're stuck in a terminal state
            if is_goal_reached(s):
                self.P[s][IDLE] = [(1.0, s, 0, True)]
            # Not a terminal state
            else:
                for a in self.available_actions(s):
                    self.P[s][a] = self.determine_probabilities(s, a, cost_function)

            it.iternext()

    # if the next state is wall, then probability of that action is zero.
    # only available actions are input
    def determine_probabilities(self, state, a, cost_function):
        """
        Determine the probabilities for the given state action set. Remind that the floor is slippery.
        :param state:   Current state number.
        :param a:       Action on the state.
        :return:        List of possible actions for the given state and action.
        """
        action_pro = []
        is_goal_reached = lambda s: s == self.goal
        adjacents = 0
        if a == UP:
            ns = state - self.max_x
            # left adjacent of next state because of the slippery floor.
            ns_l = ns-1
            # right adjacent of next state because of the slippery floor.
            ns_r = ns+1
            # then left adjacent of up is not wall.
            if self.maze.flat[ns_l] != '1':
                action_pro.append(tuple((PRO, ns_l, self.one_step_cost(state, ns_l, cost_function), is_goal_reached(ns_l))))
                adjacents += 1
            if self.maze.flat[ns_r] != '1':
                action_pro.append(tuple((PRO, ns_r, self.one_step_cost(state, ns_r, cost_function), is_goal_reached(ns_r))))
                adjacents += 1
            action_pro.append(tuple((1- adjacents*PRO, ns, self.one_step_cost(state, ns, cost_function), is_goal_reached(ns))))
        elif a == RIGHT:
            ns = state+1
            # left adjacent of next state because of the slippery floor.
            ns_l = ns-self.max_x
            # right adjacent of next state because of the slippery floor.
            ns_r = ns+self.max_x
            # then left adjacent of right is not wall.
            if self.maze.flat[ns_l] != '1':
                action_pro.append(tuple((PRO, ns_l, self.one_step_cost(state, ns_l, cost_function), is_goal_reached(ns_l))))
                adjacents += 1
            if self.maze.flat[ns_r] != '1':
                action_pro.append(tuple((PRO, ns_r, self.one_step_cost(state, ns_r, cost_function), is_goal_reached(ns_r))))
                adjacents += 1
            action_pro.append(tuple((1 - adjacents*PRO, ns, self.one_step_cost(state, ns, cost_function), is_goal_reached(ns))))

        elif a == DOWN:
            ns = state + self.max_x
            # left adjacent of next state because of the slippery floor.
            ns_l = ns+1
            # right adjacent of next state because of the slippery floor.
            ns_r = ns-1
            # then left adjacent of right is not wall.
            if self.maze.flat[ns_l] != '1':
                action_pro.append(tuple((PRO, ns_l, self.one_step_cost(state, ns_l, cost_function), is_goal_reached(ns_l))))
                adjacents += 1
            if self.maze.flat[ns_r] != '1':
                action_pro.append(tuple((PRO, ns_r, self.one_step_cost(state, ns_r, cost_function), is_goal_reached(ns_r))))
                adjacents += 1
            action_pro.append(tuple((1- adjacents*PRO, ns, self.one_step_cost(state, ns, cost_function), is_goal_reached(ns))))

        elif a == LEFT:
            ns = state - 1
            # left adjacent of next state because of the slippery floor.
            ns_l = ns + self.max_x
            # right adjacent of next state because of the slippery floor.
            ns_r = ns - self.max_x
            # then left adjacent of right is not wall.
            if self.maze.flat[ns_l] != '1':
                action_pro.append(tuple((PRO, ns_l, self.one_step_cost(state, ns_l, cost_function), is_goal_reached(ns_l))))
                adjacents += 1
            if self.maze.flat[ns_r] != '1':
                action_pro.append(tuple((PRO, ns_r, self.one_step_cost(state, ns_r, cost_function), is_goal_reached(ns_r))))
                adjacents += 1
            action_pro.append(tuple((1- adjacents*PRO, ns, self.one_step_cost(state, ns, cost_function), is_goal_reached(ns))))

        # If it is a IDLE action at the state.
        else:
            action_pro.append(tuple((1.0, state, self.one_step_cost(state, state, cost_function), is_goal_reached(state))))

        return action_pro

    def available_actions(self, state):
        """
        This function is necessary to determine the available actions for the given states.
        :param state: Current state in the maze.
        :return: List of actions that is available for that state.
        """
        actions = []
        ns_up = state - self.max_x
        ns_right = state + 1
        ns_down = state + self.max_x
        ns_left = state - 1
        # If goal state is reached
        actions.append(IDLE)
        if self.maze.flat[state] == 'G':
            return actions
        if self.maze.flat[ns_up] != '1':
            actions.append(UP)
        if self.maze.flat[ns_right] != '1':
            actions.append(RIGHT)
        if self.maze.flat[ns_down] != '1':
            actions.append(DOWN)
        if self.maze.flat[ns_left] != '1':
            actions.append(LEFT)

        return actions

    def one_step_cost(self, state, next_state, cost_function):
        """
        This function exploits two cost functions where there is no cost for transitioning into the terminal goal state.
        Also implements cost for each action.
        :param state: Current state
        :param next_state: Next state
        :return: total cost for the action
        """
        # define the initial cost as zero
        cost = 0
        # Trap affects both costs.
        if self.maze.flat[next_state] == 'T':
            cost = cost + COST_TRAP

        if cost_function == 1:
            if self.maze.flat[next_state] == 'G':
                cost = cost + GOAL_REACHED
            # For other actions, there is no cost.

        elif cost_function == 2:
            #if self.maze.flat[next_state] != 'G':
            cost = cost + COST_ENERGY

        else:
            print("Undefined cost function is selected.")

        return cost
