import problems
import GUI as gui
import copy


# import traceback
# import numpy as np

class State:
    def __init__(self, number: int, cost: int, goal: bool, actions: int):
        self.number = number  # label of this state
        self.cost = cost  # static cost taken from following any action
        self.goal = goal  # goal flag
        self.T = [[] for i in range(actions)]  # a list of successors states and probabilities following each action
        self.value = cost  # value of the state
        self.best_action = 0  # best action of the state

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

    def set_best_action(self, best_action):
        for state in self.S:
            state.best_action = best_action

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
                        # traceback.print_exc()
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

    # return an array of values and the actions
    # editables is an array of integers that correpond to the number of states
    def value_iteration(self, gamma: float, epsilon: float, editables=None):
        if not editables:  # wheter nothing was passed as editables, we will consider all states in the mdp
            editables = [s for s in self.S]
        else:
            for i in range(len(editables)):  # transforming integer into references to state's object
                if type(editables[i]) != type(self.S[0]):
                    editables[i] = self.S[editables[i] - 1]

        # value iteration
        res = float("Inf")
        vk = [0] * len(editables)
        vk1 = [0] * len(editables)
        while res > epsilon:
            aux = [[0 for s in range(len(self.S))] for a in range(self.A)]
            for s in editables:
                minimum = float('Inf')
                bestaction = 0
                for a in range(self.A):
                    summ = s.cost
                    for t in s.T[a]:
                        summ += t[1] * gamma * self.S[t[0] - 1].value
                    if summ < minimum:
                        minimum = summ
                        bestaction = a
                s.value = minimum
                s.action = bestaction

            for i in range(len(editables)):
                vk1[i] = editables[i].value
            maxi = 0
            for i in range(len(vk1)):
                summ = abs(vk1[i] - vk[i])
                if summ > maxi:
                    maxi = summ

            res = maxi
            vk = [value for value in vk1]

        return vk1

    def policy_iteration(self, gamma: float):
        pass

    def dual_criterion_risk_sensitive(self, risk_factor, minimum_error):
        # initializations
        delta1 = float('Inf')
        delta2 = 0
        v_lambda = [0] * len(self.S)
        # probability to reach the goal
        pg = [0] * len(self.S)
        for index in range(len(self.S)):
            if self.S[index].goal:
                v_lambda[index] = -1 if risk_factor > 0 else 1
                pg[index] = 1

        # auxiliar function
        def p_sum(transiction):
            summ = 0
            for s_prime in transiction:
                summ += s_prime[1] * p_previous[s_prime[0] - 1]
            return summ

        while delta1 >= minimum_error or delta2 <= 0:
            v_previous = v_lambda.copy()
            p_previous = pg.copy()
            A = [[] for i in range(len(self.S))]
            for state_index in range(len(self.S)):
                pg[state_index] = max(self.S[state_index].T, key=p_sum)
                # keeping all the actions that tie in the A list
                for transiction_index in range(len(self.S[state_index].T)):
                    if p_sum(self.S[state_index].T[transiction_index]) == pg[state_index]:
                        A[state_index].append(transiction_index)

                v_lambda[state_index] = A[state_index][0]
                best_action = 0
                for a in A[state_index]:
                    summ = 0
                    for s_prime_transaction in range(len(self.S[state_index].T[a])):
                        summ += s_prime_transaction[1] * v_previous[s_prime_transaction[0]]
                    risk_value = math.exp(risk_factor * self.S[state_index].cost) * summ
                    if risk_value > v_lambda[state_index]:
                        v_lambda[state_index] = risk_value
                        best_action = a

                self.S[state_index].action = best_action

            # update deltas
            for state_index in range(len(self.S)):
                delta1 = max(delta1, abs(v_lambda[state_index] - v_previous[state_index]) + abs(
                    pg[state_index] - p_previous[state_index]))

                all_actions = set([i for i in range(self.A)])
                max_prob_actions = set(A[state_index])
                poor_actions = all_actions - max_prob_actions
                for a in poor_actions:
                    summ = 0
                    for s_prime_transaction in range(len(self.S[state_index].T[a])):
                        summ += s_prime_transaction[1] * pg[s_prime_transaction[0]]
                    delta2 = min(delta2, pg[state_index] - summ)


def LAOStar(mdp, startState=None):
    if not startState:  # wheter nothing was passed as editables, we will consider all states in the mdp
        startState = mdp.S[0]
    else:
        if type(startState) != type(mdp.S[0]):
            startState = mdp.S[startState - 1]

    # Heuristic
    def h(state):
        # aux = mdp.S
        # mdp.value_iteration(0.999, 0.000001)
        return 0

    # LAOStar
    startState.tip = True
    G = [startState]  # Explicit graph
    while True:

        '''Expand some nonterminal tip state n of the best partial solution graph'''
        # BFS
        expanded = None
        states = [startState]
        visited = [False] * len(mdp.S)
        while states:
            s = states.pop(0)
            if s.tip:
                expanded = s
                break
            else:
                for t in s.T[s.action]:
                    state = mdp.S[t[0] - 1]
                    if not visited[state.number - 1]:
                        states.append(state)
                        visited[state.number - 1] = True

        # There are no tips
        if not expanded:
            break

        expanded.tip = False

        print("G' antes", G)
        print('Expandido:', expanded)
        '''add any new successor states to Gprime.'''
        for a in range(4):
            for t in expanded.T[a]:
                state = mdp.S[t[0] - 1]
                if state not in G:
                    G.append(state)
                    state.tip = True
                    if state.goal:
                        state.value = 0
                    else:
                        state.value = h(state)

        print("G' depois", G)

        mdp.print_actions()

        ''' Create a set Z that contains the expanded state and all of its ancestors in the explicit graph along
            marked action arcs. '''
        # DFS with path
        Z = []
        for start in G:
            states = [start]
            path = []
            visited = [False] * len(mdp.S)
            while states:
                s = states.pop()
                visited[s.number - 1] = True
                path.append(s)

                child = False

                if s == expanded:
                    for state in path:
                        if state not in Z:
                            Z.append(state)
                else:
                    for t in s.T[s.action]:
                        state = mdp.S[t[0] - 1]
                        if not visited[state.number - 1]:
                            child = True
                            states.append(state)

                if not child:
                    path.pop()
                    visited[s.number - 1] = False

        '''Perform dynamic programming on the states in set Z to update
            state costs and determine the best action for each state.'''
        print('Z', Z)
        print(mdp.value_iteration(0.999, 0.000001, Z))

        print('After update costs:')
        mdp.print_actions()
        print()


def LAOGUBS(mdp, startState=None, processed=None):
    if not startState:  # wheter nothing was passed as editables, we will consider all states in the mdp
        startState = mdp.S[0]
    else:
        if type(startState) != type(mdp.S[0]):
            startState = mdp.S[startState - 1]

    # Heuristic
    def h(state):
        # aux = mdp.S
        # mdp.value_iteration(0.999, 0.000001)
        return 0  # used with classic value_iteration to set the states's values
        # return 1 # userd with the maxprob criterion

    # LAOStar
    startState.tip = True
    G = set([startState])  # Explicit graph
    while True:

        '''
            Expand some nonterminal tip state n of the best partial solution graph
            We do a breadth-first search from the start state following the best action
            of each state until reach a tip
        '''
        # BFS
        expanded = None
        bfs = [startState]
        visited = [False] * len(mdp.S)
        while bfs:
            s = bfs.pop(0)
            if s.tip:
                expanded = s
                break
            else:
                for t in s.T[s.action]:
                    state = mdp.S[t[0] - 1]
                    if not visited[state.number - 1]:
                        bfs.append(state)
                        visited[state.number - 1] = True

        # if there are no tips the LAOStar algorithm ends
        if not expanded or expanded in processed:
            break

        # the expanded node chosen is no more a tip
        expanded.tip = False

        print("G' antes", G)
        print('Expandido:', expanded)
        '''add any new successor states to Gprime following every action.'''
        for a in range(4):

            for t in expanded.T[a]:
                state = mdp.S[t[0] - 1]
                if state not in G:
                    G.add(state)
                    state.tip = True
                    if state.goal:
                        state.value = 0
                    elif state not in processed:  # resusing the previous value of the states in the processed set
                        state.value = h(state)

        print("G' depois", G)

        mdp.print_actions()

        ''' 
            Create a set Z that contains the expanded state and all of its ancestors in the explicit graph along
            marked action arcs. 
            Here we do from every state in the explicit graph a deepth-first search keeping the path
            Then if the path from a start state x reaches the expanded state we add this path to the set Z
            At the end of this part we will have all the states in the explicit graph that can reach the expanded state
            In other words, all of its ancestors
        '''
        # DFS with path
        Z = set()
        for start in G:
            dfs = [start]
            path = []
            visited = [False] * len(mdp.S)
            while dfs:
                s = dfs.pop()
                visited[s.number - 1] = True
                path.append(s)

                child = False

                if s == expanded:
                    for state in path:
                        Z.add(state)
                else:
                    for t in s.T[s.action]:
                        state = mdp.S[t[0] - 1]
                        if not visited[state.number - 1]:
                            child = True
                            dfs.append(state)

                if not child:
                    path.pop()
                    visited[s.number - 1] = False

        '''
            Perform dynamic programming on the states in set Z to update
            state costs and determine the best action for each state.
        '''
        print('Z', Z)
        print(mdp.value_iteration(0.999, 0.000001, list(Z)))

        print('After update costs:')
        mdp.print_actions()
        print()

    '''
        Extracting the best solution graph obtained by the LAOStar algorithm.
        We keep this graph in order to reuse it if another call to this algorithm ends up
        trying to calculate a node that was already calculated by previous calls.
        We use a deepth-first search keeping the path until reach the goal state
    '''
    best_solution_graph = set()
    dfs = [startState]
    path = []
    visited = [False] * len(mdp.S)
    while dfs:
        s = dfs.pop()
        visited[s.number - 1] = True
        path.append(s)

        child = False

        if s.goal:
            for state in path:
                best_solution_graph.add(state)
        else:
            # iterate through every state reached by the current state following the best action
            for t in s.T[s.action]:
                state = mdp.S[t[0] - 1]
                if not visited[state.number - 1]:
                    child = True
                    dfs.append(state)

        # reach a tip that is not a goal state
        if not child:
            path.pop()
            visited[s.number - 1] = False
    return best_solution_graph


# Test Script
# mdp = MDP(4, 4, 4)
# # problems.swim_without_deadend(mdp.Nx,mdp.Ny,mdp.A,0.8,0,mdp)
# problems.swim(mdp.Nx, mdp.Ny, mdp.A, 0.8, 0, True, mdp)
# print(mdp)
#
# mdp.set_costs(1)
# mdp.set_action(0)
#
# # LAOStar(mdp,1)
# processed = set()
# # processed.update(LAOGUBS(mdp,12, processed))
# processed.update(LAOGUBS(mdp, 14, processed))
# print('bsg: ', processed, '\n\n\n')
# processed.update(LAOGUBS(mdp, 15, processed))
# # print('\nProcessed: ',processed,'\n\n')
# # processed.update(LAOGUBS(mdp,2, processed))
# # print('\nProcessed: ',processed,'\n\n')
#
# # print(mdp.value_iteration(0.999,0.000001))
# # mdp.print_actions()
# # print(mdp.value_iteration(0.999,0.000001,[1,2,3]))
# # print(mdp.value_iteration(0.999,0.000001,[1,2,3,5]))

mdp = MDP(4, 10, 4)
problems.swim_symmetric(mdp.Nx, mdp.Ny, mdp.A, 0.5, 0, True, mdp)