import math
import copy
# import traceback
# import numpy as np

class State:
    def __init__(self, number: int, cost: int, goal: bool):
        self.number = number
        self.cost = cost
        self.goal = goal
        self.T = [[] for i in range(4)]  # a set of states and a set of probabilities for each action
        self.value = cost #value at that moment
        self.action = 0 #best action at that moment
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
            self.S.append(State(num + 1, 1, False))
        goal_state = (Nx - 1) * Ny
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
                    s += str(num).ljust(12,' ')
                s += '\n'
            s += '\n'
            k += 1
        s += "A = " + str(self.A) + "\nS = " + str(self.S) \
             + "\nNx = " + str(self.Nx) + "\nNy = " + str(self.Ny)\
             + "\ngoal_vector = " + str(self.goal_vector()) + "\ncost_vector = " + str(self.cost_vector()) + "\n"
        return s


    def set_costs(self, cost):
        for state in self.S:
            state.cost = cost

    def set_action(self, action):
        for state in self.S:
            state.action = action

    def print_actions(self):
        for i in range(self.Ny - 1, -1, -1):
            for j in range(i, i + len(self.S) - self.Ny, self.Ny):
                print(self.S[j].action, end=' ')
            print()
        print()

    def add_edge(self, s1: int, a: int, s2: list, p: list):
        #adjusting arguments
        if type(s1) != type(self.S[0]):
            s1 = self.S[s1-1]
        if type(s2) != type([]):
            s2 = [s2]
        for i in range(len(s2)):
            if type(s2[i]) == type(self.S[0]):
                s2[i] = s2[i].number

        #adding edge
        for s in s2:
            s1.T[a].append([s,p])

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
        sub = MDP(1,1)
        sub.S = []
        if type(state) != type(self.S[0]):
            state = self.S[state-1]
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
                        self.S[s1].T[a].append([s2+1,T[a][s1][s2]])

    def iterations_repr(self, arr):
        v = 'Values\n'
        a = 'Actions\n'
        for i in range(len(arr)-1,-1,-1):
            print(i)
        #     for line in a:
        #         for num in line:
        #             s += str(num).ljust(12,' ')
        #         s += '\n'
        #     s += '\n'
        #     k += 1
        # s += "A = " + str(self.A) + "\nS = " + str(self.S) \
        #      + "\nNx = " + str(self.Nx) + "\nNy = " + str(self.Ny)\
        #      + "\ngoal_vector = " + str(self.goal_vector()) + "\ncost_vector = " + str(self.cost_vector()) + "\n"

    #return an array of values and the actions
    #editables is an array of integers that correpond to the number of states
    def value_iteration(self,gamma: float, epsilon: float, editables=None):
        if not editables: #wheter nothing was passed as editables, we will consider all states in the mdp
            editables = [s for s in self.S]
        else:
            for i in range(len(editables)): #transforming integer into references to state's object
                if type(editables[i]) != type(self.S[0]):
                    editables[i] = self.S[editables[i]-1]

        #value iteration
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
                        summ += t[1] * gamma * self.S[t[0]-1].value
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

    def policy_iteration(self,gamma: float):
        pass

    def swim(self, probCorrenteza, probFall, Bridge):
        T = [[[0 for s2 in range(len(self.S))] for s1 in range(len(self.S))] for a in range(4)]

        for s in range(self.Nx * self.Ny):  # each state
            y = s % self.Ny;
            x = math.floor(s / self.Ny);

            if y == 0:  # goal or waterfall
                if x == self.Nx - 1:  # goal
                    for a in range(self.A):
                        T[a][s][len(
                            self.S) - 1] = 1;  # for each action it has 1 of probability to go to a absorb state where costs are zero
                else:
                    for a in range(self.A):
                        T[a][s][s] = 1;  # for each action it has 1 of probability to go to itself
            else:
                # UP
                a = 0
                if x > 0 and x < self.Nx - 1 and (y < self.Ny - 1 or not Bridge):  # rio
                    x1 = x;
                    x2 = x;
                    x3 = x;
                    y1 = min(y + 1, self.Ny - 1);
                    y2 = y - 1;
                    y3 = y;

                    s1 = x1 * self.Ny + y1;  # sobe
                    s2 = x2 * self.Ny + y2;  # desce
                    s3 = x3 * self.Ny + y3;  # parado

                    T[a][s][s1] = (1 - probCorrenteza) ** 2;
                    T[a][s][s2] = T[a][s][s2] + probCorrenteza ** 2;
                    T[a][s][s3] = T[a][s][s3] + 2 * probCorrenteza * (1 - probCorrenteza);
                else:  # margem
                    x1 = x;
                    y1 = min(y + 1, self.Ny - 1);
                    s1 = x1 * self.Ny + y1;
                    x2 = min(x + 1, self.Nx - 1);
                    y2 = y;
                    s2 = x2 * self.Ny + y2;
                    T[a][s][s1] = 1 - probFall;
                    T[a][s][s2] = T[a][s][s2] + probFall;

                # DOWN
                a = 1
                x1 = x;
                y1 = y - 1;
                s1 = x1 * self.Ny + y1;
                T[a][s][s1] = 1;

                # RIGHT
                a = 2
                if x > 0 and x < self.Nx - 1 and (y < self.Ny - 1 or not Bridge):  # rio
                    x1 = min(x + 1, self.Nx - 1);
                    x2 = x;
                    x3 = min(x + 1, self.Nx - 1);
                    x4 = x;
                    y1 = y;
                    y2 = y - 1;
                    y3 = y - 1;
                    y4 = y;
                    s1 = x1 * self.Ny + y1;  # east
                    s2 = x2 * self.Ny + y2;  # south
                    s3 = x3 * self.Ny + y3;  # southeast
                    s4 = x4 * self.Ny + y4;  # stopped

                    T[a][s][s1] = (1 - probCorrenteza) ** 2;
                    T[a][s][s2] = T[a][s][s2] + probCorrenteza ** 2;
                    T[a][s][s3] = T[a][s][s3] + (1 - probCorrenteza) * probCorrenteza;
                    T[a][s][s4] = T[a][s][s4] + probCorrenteza * (1 - probCorrenteza);
                else:
                    x1 = min(x + 1, self.Nx - 1);
                    y1 = y;
                    s1 = x1 * self.Ny + y1;
                    T[a][s][s1] = 1

                # LEFT
                a = 3
                if x > 0 and x < self.Nx - 1 and (y < self.Ny - 1 or not Bridge):  # river
                    x1 = max(x - 1, 0);
                    x2 = x;
                    x3 = max(x - 1, 0);
                    x4 = x;
                    y1 = y;
                    y2 = y - 1;
                    y3 = y - 1;
                    y4 = y;
                    s1 = x1 * self.Ny + y1;  # weast
                    s2 = x2 * self.Ny + y2;  # south
                    s3 = x3 * self.Ny + y3;  # southwest
                    s4 = x4 * self.Ny + y4;  # stopped
                    T[a][s][s1] = (1 - probCorrenteza) ** 2;
                    T[a][s][s2] = T[a][s][s2] + probCorrenteza ** 2;
                    T[a][s][s3] = T[a][s][s3] + (1 - probCorrenteza) * probCorrenteza;
                    T[a][s][s4] = T[a][s][s4] + probCorrenteza * (1 - probCorrenteza);
                else:
                    x1 = max(x - 1, 0);
                    y1 = y;
                    s1 = x1 * self.Ny + y1;
                    T[a][s][s1] = 1;

        for a in range(self.A):
            for s2 in range(len(self.S)):
                T[a][(self.Nx - 1) * self.Ny][s2] = 0
            T[a][(self.Nx - 1) * self.Ny][len(self.S) - 1] = 1

        self.matrix_to_edges(T)
        return T


def LAOStar(mdp, startState=None):
    if not startState:  # wheter nothing was passed as editables, we will consider all states in the mdp
        startState = mdp.S[0]
    else:
        if type(startState) != type(mdp.S[0]):
            startState = mdp.S[startState-1]

    #Heuristic
    def h(state):
        # aux = mdp.S
        # mdp.value_iteration(0.999, 0.000001)
        return 0

    #LAOStar
    startState.tip = True
    G = [startState] #Explicit graph
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
                    state = mdp.S[t[0]-1]
                    if not visited[state.number-1]:
                        states.append(state)
                        visited[state.number-1] = True

        # There are no tips
        if not expanded:
            break

        expanded.tip = False

        print("G' antes", G)
        print('Expandido:', expanded)
        '''add any new successor states to Gline.'''
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
                visited[s.number-1] = True
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
                    visited[s.number-1] = False

        '''Perform dynamic programming on the states in set Z to update
            state costs and determine the best action for each state.'''
        print('Z',Z)
        print(mdp.value_iteration(0.999,0.000001,Z))

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
        return 0 # used with classic value_iteration to set the states's values
        # return 1 # userd with the maxprob criterion

    # LAOStar
    startState.tip = True
    G = [startState]  # Explicit graph
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
        '''add any new successor states to Gline following every action.'''
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
mdp = MDP(4,4)
mdp.swim(0.8,0,True)
print(mdp)

mdp.set_costs(1)
mdp.set_action(0)

# LAOStar(mdp,1)
processed = set()
processed.update(LAOGUBS(mdp,12, processed))
print('\nProcessed: ',processed,'\n\n')
processed.update(LAOGUBS(mdp,2, processed))
print('\nProcessed: ',processed,'\n\n')


# print(mdp.value_iteration(0.999,0.000001))
# mdp.print_actions()
# print(mdp.value_iteration(0.999,0.000001,[1,2,3]))
# print(mdp.value_iteration(0.999,0.000001,[1,2,3,5]))
