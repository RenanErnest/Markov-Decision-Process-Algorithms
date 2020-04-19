import math
import problems
import GUI as gui


class State:
    def __init__(self, number: int, cost: int, goal: bool, actions: int):
        self.number = number  # label of this state
        self.cost = cost  # static cost taken from following any action
        self.goal = goal  # goal flag
        self.T = [[] for i in range(actions)]  # a list of successors states and probabilities following each action

    def __str__(self):
        return str(self.number)

    def __repr__(self):
        return str(self.number)


class MDP:
    def __init__(self, Nx, Ny, actions):
        self.A = actions  # number of actions
        self.S = [State(index + 1, 1, False, actions) for index in range(Nx * Ny + 1)]  # list with all states
        self.Nx = Nx
        self.Ny = Ny

    def __repr__(self):
        s = ""
        T = self.transition_matrix()
        directions = ['UP', 'DOWN', 'RIGHT', 'LEFT']
        k = 0
        for a in T:
            s += 'T[' + str(k) + '] = ' + directions[k] + '\n'
            for line in a:
                for num in line:
                    s += str(num).ljust(12, ' ')
                s += '\n'
            s += '\n'
            k += 1
        s += "A = " + str(self.A) + "\nS = " + str(self.S) \
             + "\nNx = " + str(self.Nx) + "\nNy = " + str(self.Ny) \
             + "\ngoal_vector = " + str(self.goal_vector()) + "\ncost_vector = " + str(self.cost_vector()) + "\n"
        return s

    def set_costs(self, cost):
        for state in self.S:
            state.cost = cost

    def print_actions(self):
        for i in range(self.Ny - 1, -1, -1):
            for j in range(i, i + len(self.S) - self.Ny, self.Ny):
                print(self.S[j].best_action, end=' ')
            print()
        print()

    def add_edge(self, s1: int, a: int, s2: list, p: list):
        # adjusting arguments
        if type(s1) != type(self.S[0]):
            s1 = self.S[s1 - 1]
        if type(s2) != type([]):
            s2 = [s2]
        for i in range(len(s2)):
            if type(s2[i]) == type(self.S[0]):
                s2[i] = s2[i].number

        # adding edge
        for s in s2:
            s1.T[a].append({'state': s, 'prob': p})

    def transition_matrix(self):
        T = [[[0 for s2 in range(len(self.S))] for s1 in range(len(self.S))] for a in range(4)]
        for state in self.S:
            for action in range(self.A):
                for t in state.T[action]:
                    try:
                        aux = None
                        for s in self.S:
                            if s.number == t['state']:
                                aux = s
                                break
                        T[action][self.S.index(state)][self.S.index(aux)] = t['prob']
                    except:
                        pass
        return T

    def cost_vector(self):
        c = []
        for state in self.S:
            c.append(state.cost)
        return c

    def goal_vector(self):
        g = []
        for state in self.S:
            g.append(int(state.goal))
        return g

    def matrix_to_edges(self, T: list):
        for a in range(len(T)):
            for s1 in range(len(T[a])):
                for s2 in range(len(T[a][s1])):
                    if T[a][s1][s2] > 0:
                        self.S[s1].T[a].append({'state': s2 + 1, 'prob': T[a][s1][s2]})

    def iterations_repr(self, arr):
        v = 'Values\n'
        a = 'Actions\n'
        for i in range(len(arr) - 1, -1, -1):
            print(i)


def value_iteration(mdp, gamma=0.999, epsilon=0.00001):
    """
    update the values of the state's object and also returns a list with the values of each state
    :param mdp: a MDP object
    :param gamma: used for limit the propagation of infinite values
    :param epsilon: precision measured by the number of zeros on epsilon
    :return: list of values for each state as well as the best actions that gave the minimum values of each state
    """
    res = float("Inf")
    vk = [0] * len(mdp.S)  # values for each state in the mdp
    vk1 = [0] * len(mdp.S)
    best_actions = [0] * len(mdp.S)
    while res > epsilon:
        for s in mdp.S:  # for each state in the mdp
            minimum = float('Inf')
            best_action = 0
            for a in range(mdp.A):  # following each action
                #  summation
                summ = s.cost
                for t in s.T[a]:  # for each successor states and probabilities
                    summ += t['prob'] * gamma * vk[t['state'] - 1]

                if summ < minimum:
                    minimum = summ
                    best_action = a

            vk1[s.number - 1] = minimum
            best_actions[s.number - 1] = best_action

        maxi = 0
        for i in range(len(vk1)):
            summ = abs(vk1[i] - vk[i])
            if summ > maxi:
                maxi = summ
        res = maxi
        vk = vk1.copy()

    return vk1, best_actions


def dual_criterion_risk_sensitive(mdp, risk_factor=-0.01, minimum_error=0.0001):
    """
    update the values of the state's object and also returns a list with the values of each state
    :param mdp: a MDP object
    :param risk_factor: used for weight the risk value
    :param minimum_error: precision measured by the number of zeros on epsilon
    :return: a list of probability to goal values, a list of risk values, and a list of best actions, both for each state
    """
    # initializations
    delta1 = float('Inf')
    delta2 = 0
    v_lambda = [0] * len(mdp.S)  # risk values
    pg = [0] * len(mdp.S)  # probability to reach the goal
    best_actions = [0] * len(mdp.S)  # best policy that gave us the best values

    for s in mdp.S:  # goal states treatment
        if s.goal:
            v_lambda[s.number - 1] = -1 if risk_factor > 0 else 1  # -sgn(risk_factor)
            pg[s.number - 1] = 1

    def linear_combination(t, vector):
        """
        an auxiliar function in order to clear the code
        :param t: a transaction list obtained by following an action on a state, usually mdp.S[i].T[a]
        :param vector: a vector whose will be linear combined with the states and probabilities of t
        :return: a summation representing the linear combination
        """
        summ = 0
        for s2 in t:
            summ += s2['prob'] * vector[s2['state'] - 1]
        return summ

    while delta1 >= minimum_error or delta2 <= 0:
        v_previous = v_lambda.copy()
        p_previous = pg.copy()

        A = [[] for i in range(len(mdp.S))]  # max prob actions for each state

        for s in mdp.S:

            if s.goal:
                continue

            # probability value section
            max_prob_t = max(s.T, key=lambda t: linear_combination(t, p_previous))
            pg[s.number - 1] = linear_combination(max_prob_t, p_previous)

            # keeping all the actions that tie with the max prob action in the A list of this state
            A[s.number - 1] = []
            for t_index in range(len(s.T)):  # the indexes represent the taken actions
                if linear_combination(s.T[t_index], p_previous) == pg[s.number - 1]:
                    A[s.number - 1].append(t_index)

            # risk value section
            best_action = max(A[s.number - 1], key=lambda a: math.exp(risk_factor * s.cost) *
                                                                        linear_combination(s.T[a], v_previous))
            v_lambda[s.number - 1] = math.exp(risk_factor * s.cost) * linear_combination(s.T[best_action], v_previous)
            best_actions[s.number-1] = best_action


        # updating deltas
        n_delta1 = -float('Inf')
        n_delta2 = float('Inf')
        for s in mdp.S:

            if s.goal:
                continue

            n_delta1 = max(n_delta1, abs(v_lambda[s.number-1] - v_previous[s.number-1]) + abs(
                pg[s.number-1] - p_previous[s.number-1]))

            all_actions = set([i for i in range(mdp.A)])
            max_prob_actions = set(A[s.number-1])
            poor_actions = all_actions - max_prob_actions
            for a in poor_actions:
                n_delta2 = min(n_delta2, pg[s.number-1] - linear_combination(s.T[a], pg))

        delta1 = n_delta1
        ''' if there will be no poor actions, delta2 assumes the best value, Inf'''
        delta2 = n_delta2


    return pg,v_lambda,best_actions


# Test Script
# mdp = MDP(4, 8, 4)
# problems.swim_symmetric(mdp.Nx, mdp.Ny, mdp.A, 0.8, 0, True, mdp)
# print(mdp)
# mdp.set_costs(1)
#
# # Costs value iteration
# (cost_values,best_actions) = value_iteration(mdp, 0.999, 0.0001)
# gui.plot(mdp,[cost_values],['V'],best_actions)
#
# # Dual criterion
# (prob_to_goal,risk_values,best_actions) = dual_criterion_risk_sensitive(mdp, -0.01, 0.0001)
# gui.plot(mdp,[prob_to_goal,risk_values],['PG','V'],best_actions)