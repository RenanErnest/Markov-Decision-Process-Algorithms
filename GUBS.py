import problems
import copy
import gubsGUI as gui
import math


# import traceback
# import numpy as np

class State:
    def __init__(self, number: int, cost: int, goal: bool, actions: int):
        self.number = number
        self.cost = cost
        self.goal = goal
        self.T = [[] for i in range(actions)]  # a set of states and a set of probabilities for each action
        self.value = cost  # whatever
        self.probReachGoal = 0
        self.action = 0  # best action at that moment
        self.tip = goal

    def __str__(self):
        return str(self.number)

    def __repr__(self):
        return str(self.number)


class MDP:
    def __init__(self, Nx, Ny):
        self.A = 4
        self.S = []
        for num in range(Nx * Ny + 1):
            self.S.append(State(num + 1, 1, False,self.A))
        # goal_state = (Nx - 1) * Ny
        goal_state = Nx*Ny
        self.S[goal_state].cost = self.S[Nx * Ny].cost = 0
        self.S[goal_state].goal = True
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

    def set_action(self, action):
        for state in self.S:
            state.action = action

    def set_value(self, value):
        for state in self.S:
            state.value = value

    def print_actions(self):
        for i in range(self.Ny - 1, -1, -1):
            for j in range(i, i + len(self.S) - self.Ny, self.Ny):
                print(self.S[j].action, end=' ')
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
            s1.T[a].append([s, p])

    def transition_matrix(self):
        T = [[[0 for s2 in range(len(self.S))] for s1 in range(len(self.S))] for a in range(4)]
        for state in self.S:
            for action in range(self.A):
                for t in state.T[action]:
                    try:
                        aux = None
                        for s in self.S:
                            if s.number == t[0]:
                                aux = s
                                break
                        T[action][self.S.index(state)][self.S.index(aux)] = t[1]
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

    # it creates a graph with the state parameter and its ancestors
    def subgraph_with_ancestors(self, state):
        sub = MDP(1, 1)
        sub.S = []
        if type(state) != type(self.S[0]):
            state = self.S[state - 1]
        sub.S.append(copy.deepcopy(state))
        for s in self.S:
            for a in range(self.A):
                if s.T[a][0].count(state) > 0:
                    same = False
                    for substate in sub.S:
                        if substate.number == s.number:
                            same = True
                    if not same:
                        sub.S.append(copy.deepcopy(s))
        sub.S.sort(key=lambda x: x.number)

        # adjusting references
        for s in range(len(sub.S)):
            for a in range(len(sub.S[s].T)):
                for s2 in range(len(sub.S[s].T[a][0])):
                    for s1 in sub.S:
                        if sub.S[s].T[a][0][s2].number == s1.number:
                            sub.S[s].T[a][0][s2] = s1
        return sub

    def matrix_to_edges(self, T: list):
        for a in range(len(T)):
            for s1 in range(len(T[a])):
                for s2 in range(len(T[a][s1])):
                    if T[a][s1][s2] > 0:
                        self.S[s1].T[a].append([s2 + 1, T[a][s1][s2]])

    def iterations_repr(self, arr):
        v = 'Values\n'
        a = 'Actions\n'
        for i in range(len(arr) - 1, -1, -1):
            print(i)

    # return an array of values and the actions
    # Z is an array of integers that correpond to the number of states
    def value_iteration(self, gamma: float, epsilon: float, Z=None):
        if not Z:  # wheter nothing was passed as Z, we will consider all states in the mdp
            Z = [s for s in self.S]

        # value iteration
        res = float("Inf")
        vk = [0] * len(Z)
        vk1 = [0] * len(Z)

        while res > epsilon:
            aux = [[0 for s in range(len(self.S))] for a in range(self.A)]
            for s in Z:
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

            for i in range(len(Z)):
                vk1[i] = Z[i].value
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

    def dual_criterion_risk_sensitive(self, risk_factor, minimum_error, Z=None):
        if not Z:  # wheter nothing was passed as Z, we will consider all states in the mdp
            Z = [s for s in self.S]

        delta1 = float('Inf')
        delta2 = 0
        # probability to reach the goal
        pg = [s.probReachGoal for s in self.S]
        v_lambda = [s.value for s in self.S]

        for s in Z:
            pg[s.number-1] = 0
            v_lambda[s.number-1] = -1 if risk_factor > 0 else 1

        for s in self.S: #goals
            if s.goal:
                v_lambda[s.number-1] = -1 if risk_factor > 0 else 1
                pg[s.number-1] = 1

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
            for state in Z:
                if state.goal:
                    state.probReachGoal = pg[state.number - 1]
                    state.value = v_lambda[state.number - 1]
                    continue
                A[state.number-1] = []
                pg[state.number-1] = p_sum(max(state.T, key=p_sum))
                # keeping all the actions that tie in the A list
                for transiction_index in range(len(state.T)):
                    if p_sum(state.T[transiction_index]) == pg[state.number-1]:
                        A[state.number-1].append(transiction_index)

                max_v_lambda = -float('Inf')
                best_action = 0
                for a in A[state.number-1]:
                    summ = 0
                    for s_prime_transaction in state.T[a]:
                        summ += s_prime_transaction[1] * v_previous[s_prime_transaction[0]-1]
                    risk_value = math.exp(risk_factor * state.cost) * summ
                    if risk_value > max_v_lambda:
                        max_v_lambda = risk_value
                        best_action = a
                v_lambda[state.number-1] = max_v_lambda

                # updating best action and values
                state.probReachGoal = pg[state.number-1]
                state.value = v_lambda[state.number-1]
                state.action = best_action

            # update deltas
            delta1 = abs(v_lambda[0] - v_previous[0]) + abs(pg[0] - p_previous[0])
            for state_index in range(len(self.S)):
                if self.S[state_index].goal:
                    continue

                delta1 = max(delta1,abs(v_lambda[state_index] - v_previous[state_index]) + abs(
                    pg[state_index] - p_previous[state_index]))

            n_delta2 = float('Inf')
            all_actions = set([i for i in range(self.A)])
            for state in Z:
                if state.goal:
                    continue
                max_prob_actions = set(A[state.number-1])
                poor_actions = all_actions - max_prob_actions
                for a in poor_actions:
                    summ = 0
                    for s_prime_transaction in state.T[a]:
                        summ += s_prime_transaction[1] * pg[s_prime_transaction[0] - 1]
                    n_delta2 = min(n_delta2, pg[state.number-1] - summ)
            delta2 = n_delta2
            print(delta1, delta2)


def LAOGUBS(mdp, risk_factor, error_minimum, startState=None, processed=None):
    if not startState:
        startState = mdp.S[0]
    else:
        if type(startState) != type(mdp.S[0]):
            startState = mdp.S[startState - 1]

    # Heuristic
    def h(state):
        # aux = mdp.S
        # mdp.value_iteration(0.999, 0.000001)
        return 1

    # LAOStar
    startState.tip = True
    G = set([startState])  # Explicit graph

    def expand():
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
            return None

        # the expanded node chosen is no more a tip
        expanded.tip = False

        return expanded

    def sucessors(expanded,G):
        '''add any new successor states to Gprime following every action.'''
        for a in range(4):

            for t in expanded.T[a]:
                state = mdp.S[t[0] - 1]
                if state not in G:
                    G.add(state)
                    state.tip = True
                    if state.goal:
                        state.probReachGoal = 1
                    elif state not in processed:  # reusing the previous value of the states in the processed set
                        state.probReachGoal = h(state)
                        state.value = -1 if risk_factor > 0 else 1 # -sgn(risk_factor)
        return G

    def setZ(expanded, G):
        '''
            Create a set Z that contains the expanded state and all of its ancestors in the explicit graph along
            marked action arcs.
            Here we do from every state in the explicit graph a recursion keeping the path
            Then if the path from a start state x reaches the expanded state we add this path to the set Z
            At the end of this part we will have all the states in the explicit graph that can reach the expanded state
            In other words, all of its ancestors
        '''

        def recursion(s,path,visited,Z):
            visited[s.number - 1] = True
            path.append(s)
            child = False

            if s == expanded:
                for state in path:
                    if not state.tip:
                        Z.add(state)
            else:
                for t in s.T[s.action]:
                    state = mdp.S[t[0] - 1]
                    if not visited[state.number - 1] and state in G:
                        if not child:
                            child = recursion(state,path,visited,Z)

            if not child:
                path.pop()
                visited[s.number - 1] = False

            return child

        # DFS with path
        Z = set()
        for start in G:
            path = []
            visited = [False] * len(mdp.S)
            recursion(start, path, visited,Z)
        return Z

    def update(Z):
        '''
            Perform dynamic programming on the states in set Z to update
        state costs and determine the best action for each state.
        '''
        # mdp.value_iteration(0.999, 0.000001, list(Z))
        mdp.dual_criterion_risk_sensitive(-0.01,0.001, list(Z))

    gui.plot(mdp, [expand,sucessors,setZ,update])

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


# x = int(input('Quantidade de estados no eixo X: '))
# y = int(input('Quantidade de estados no eixo Y: '))
# probFall = float(input('Probabilidade de ser levado pela correnteza: '))

# Test Script
mdp = MDP(8, 6)
# problems.swim_without_deadend(mdp.Nx,mdp.Ny,mdp.A,0.8,0,mdp)
#problems.swim(mdp.Nx, mdp.Ny, mdp.A, 0.8, 0, True, mdp)
problems.swim_symmetric(mdp.Nx, mdp.Ny, mdp.A, 0.8, 0, True, mdp)
mdp = MDP(4, 8)
problems.swim_symmetric(mdp.Nx, mdp.Ny, mdp.A, 0.8, 0, True, mdp)
print(mdp)

mdp.set_costs(1)
mdp.set_action(0)

# GUBS
risk_factor = -0.01
error_minimum = 0.0001
value = 0

mdp.set_value(value)

mdp.dual_criterion_risk_sensitive(risk_factor,error_minimum)

# mdp.dual_criterion_risk_sensitive(risk_factor,error_minimum,[mdp.S[3],mdp.S[7],mdp.S[11],mdp.S[15]])
# mdp.dual_criterion_risk_sensitive(risk_factor,error_minimum,[mdp.S[10],mdp.S[7],mdp.S[11],mdp.S[15]])
# mdp.dual_criterion_risk_sensitive(risk_factor,error_minimum,[mdp.S[15],mdp.S[14],mdp.S[13],mdp.S[12]])
gui.plot(mdp, [])

processed = set()
# processed.update(LAOGUBS(mdp, 14, processed))
# print('bsg: ', processed, '\n\n\n')
# processed.update(LAOGUBS(mdp, 15, processed))
# print('\nProcessed: ',processed,'\n\n')
# processed.update(LAOGUBS(mdp,29, processed))
# print('\nProcessed: ',processed,'\n\n')

