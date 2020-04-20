import math

import MDP


def LAO(mdp, start_state=1):
    '''
    :param mdp: an MDP object
    :param start_state: an integer representing the start state
    :return: the final values and best actions determined by the dynamic programming function

    In order to change the dynamic programming function you have to pay a special attention to:
        - an admissible heuristic values
        - starter values
        - goal values assigned on successors have to be the best values according to the dynamic programming algorithm used
        - the dynamic programming algorithm has to keep values and best actions through calls
        - the return of LAO
    All of this changes are marked with a comment '# *' on the code
    '''

    # Heuristic
    def h(s):  # *
        return 0

    best_actions = [0] * len(mdp.S)
    values = [0] * len(mdp.S)  # *

    start_state = mdp.S[start_state - 1]
    tip = [False] * len(mdp.S)
    tip[start_state.number - 1] = True
    G = set([start_state])  # Explicit graph

    while True:

        '''
            Expand some nonterminal tip state n of the best partial solution graph
            We do a breadth-first search from the start state following the best actions until reach a tip
        '''
        # BFS
        expanded = None
        bfs = [start_state]
        visited = [False] * len(mdp.S)
        while bfs:
            s = bfs.pop(0)
            if tip[s.number - 1]:
                expanded = s
                break
            else:
                for t in s.T[best_actions[s.number - 1]]:
                    s2 = mdp.S[t['state'] - 1]
                    if not visited[s2.number - 1]:
                        bfs.append(s2)
                        visited[s2.number - 1] = True

        # If there are no tips to expand the LAO algorithm is over
        if not expanded:
            break

        # The expanded node chosen is no more a tip
        tip[expanded.number - 1] = False

        '''add any new successor states to G following any action from the expanded state.'''
        for a in range(mdp.A):
            for t in expanded.T[a]:
                s2 = mdp.S[t['state'] - 1]
                if s2 not in G:
                    G.add(s2)
                    tip[s2.number - 1] = True
                    if s2.goal:  # *
                        # this value has to be the best value according the dynamic programming algorithm used
                        values[s2.number - 1] = 0
                    else:
                        values[s2.number - 1] = h(s2)

        '''
            Create a set Z that contains the expanded state and all of its ancestors in the explicit graph along
            marked action arcs.
            Here we do from every state in the explicit graph a recursion keeping the path
            Then if the path from a start state x reaches the expanded state we add this path to the set Z
            At the end of this part we will have all the states in the explicit graph that can reach the expanded state
            In other words, all of its ancestors
        '''
        def recursion(s,visited,Z):
            visited[s.number-1] = True

            found = False

            if s == expanded:
                Z.add(s)  # Z is a set so there is no repetitions
                return True

            for t in s.T[best_actions[s.number-1]]:
                s2 = mdp.S[t['state']-1]
                # if s2 is in G and it is not a tip and it was not visited
                if s2 in G and not tip[s2.number - 1] and not visited[s2.number - 1]:
                    if not found:
                        found = recursion(s2,visited,Z)
                    else:
                        recursion(s2,visited,Z)

            if found:
                Z.add(s)
                return True
            return False

        Z = set()
        for start in G:
            if not tip[start.number-1]:
                visited = [False] * len(mdp.S)
                recursion(start, visited, Z)

        '''
            Perform dynamic programming on the states in set Z in order to update values and best actions
        '''
        (values, best_actions) = z_value_iteration(mdp, Z, values, best_actions, 0.999, 0.000001)  # *

    return values, best_actions  # *


def z_value_iteration(mdp, Z, values, best_actions, gamma=0.999, epsilon=0.00001):
    """
    update the values of the state's object and also returns a list with the values of each state
    :param mdp: a MDP object
    :param Z: a set of states that will be updated
    :param values: previous values of each state
    :param best_actions: previous best actions of each state
    :param gamma: used for limit the propagation of infinite values
    :param epsilon: precision measured by the number of zeros on epsilon
    :return: list of values for each state as well as the best actions that gave the minimum values of each state
    """
    if not Z:  # whether nothing was passed as Z, we will consider all states in the mdp
        Z = [s for s in mdp.S]

    res = float("Inf")
    vk = values.copy()
    vk1 = vk.copy()
    while res > epsilon:
        for s in Z:  # for each state in the mdp
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


def z_dual_criterion_risk_sensitive(mdp, Z, pg_values, risk_values, best_actions, risk_factor=-0.01, minimum_error=0.0001):
    """
    update the values of the state's object and also returns a list with the values of each state
    :param mdp: a MDP object
    :param Z: a set of states that will be updated
    :param pg_values: previous probability to goal values of each state
    :param risk_values: previous risk values of each state
    :param best_actions: previous best actions of each state
    :param risk_factor: used for weight the risk value
    :param minimum_error: precision measured by the number of zeros on epsilon
    :return: list of pg_values and risk_values for each state as well as the best actions that gave the best values
    """
    if not Z:  # whether nothing was passed as Z, we will consider all states in the mdp
        Z = [s for s in mdp.S]

    # initializations
    delta1 = float('Inf')
    delta2 = 0
    v_lambda = risk_values.copy()  # risk values
    pg = pg_values.copy()  # probability to reach the goal

    for s in Z:
        pg[s.number - 1] = 0

    for s in Z:  # goal states treatment
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

        for s in Z:

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
            best_actions[s.number - 1] = best_action

        # updating deltas
        n_delta1 = -float('Inf')
        n_delta2 = float('Inf')
        for s in Z:

            if s.goal:
                continue

            n_delta1 = max(n_delta1, abs(v_lambda[s.number - 1] - v_previous[s.number - 1]) + abs(
                pg[s.number - 1] - p_previous[s.number - 1]))

            all_actions = set([i for i in range(mdp.A)])
            max_prob_actions = set(A[s.number - 1])
            poor_actions = all_actions - max_prob_actions
            for a in poor_actions:
                n_delta2 = min(n_delta2, pg[s.number - 1] - linear_combination(s.T[a], pg))

        delta1 = n_delta1
        ''' if there will be no poor actions, delta2 assumes the best value, Inf'''
        delta2 = n_delta2

    return pg, v_lambda, best_actions


def LAO_interactive(mdp, start_state=1):
    """
    The same as the LAO function above but divided into functions in order to visualize each step on GUI

    :param mdp: an MDP object
    :param start_state: an integer representing the start state

    In order to change the dynamic programming function you have to pay a special attention to:
        - an admissible heuristic values
        - starter values
        - goal values assigned on successors have to be the best values according to the dynamic programming algorithm used
        - the dynamic programming algorithm has to keep values and best actions through calls
    All of this changes are marked with a comment '# *' on the code
    """

    # Heuristic
    def h(s):  # *
        return 0

    def expand(start_state,best_actions,tip):
        '''
            Expand some nonterminal tip state n of the best partial solution graph
            We do a breadth-first search from the start state following the best actions until reach a tip
        '''

        # BFS
        expanded = None
        bfs = [start_state]
        visited = [False] * len(mdp.S)
        while bfs:
            s = bfs.pop(0)
            if tip[s.number - 1]:
                expanded = s
                break
            else:
                for t in s.T[best_actions[s.number - 1]]:
                    s2 = mdp.S[t['state'] - 1]
                    if not visited[s2.number - 1]:
                        bfs.append(s2)
                        visited[s2.number - 1] = True

        # If there are no tips to expand the LAO algorithm is over
        if not expanded:
            return None

        # The expanded node chosen is no more a tip
        tip[expanded.number - 1] = False

        return expanded

    def successors(expanded, G,tip,values):
        '''add any new successor states to G following any action from the expanded state.'''
        for a in range(mdp.A):
            for t in expanded.T[a]:
                s2 = mdp.S[t['state'] - 1]
                if s2 not in G:
                    G.add(s2)
                    tip[s2.number - 1] = True
                    if s2.goal:  # *
                        # this value has to be the best value according the dynamic programming algorithm used
                        values[s2.number - 1] = 0
                    else:
                        values[s2.number - 1] = h(s2)
        return G

    def setZ(expanded, G, tip, best_actions):
        '''
            Create a set Z that contains the expanded state and all of its ancestors in the explicit graph along
            marked action arcs.
            Here we do from every state in the explicit graph a recursion keeping the path
            Then if the path from a start state x reaches the expanded state we add this path to the set Z
            At the end of this part we will have all the states in the explicit graph that can reach the expanded state
            In other words, all of its ancestors
        '''
        def recursion(s,visited,Z):
            visited[s.number-1] = True

            found = False

            if s == expanded:
                Z.add(s)  # Z is a set so there is no repetitions
                return True

            for t in s.T[best_actions[s.number-1]]:
                s2 = mdp.S[t['state']-1]
                # if s2 is in G and it is not a tip and it was not visited
                if s2 in G and not tip[s2.number - 1] and not visited[s2.number - 1]:
                    if not found:
                        found = recursion(s2,visited,Z)
                    else:
                        recursion(s2,visited,Z)

            if found:
                Z.add(s)
                return True
            return False

        Z = set()
        for start in G:
            if not tip[start.number-1]:
                visited = [False] * len(mdp.S)
                recursion(start, visited, Z)

        return Z

    def dynamic_programming(Z,values,best_actions):
        '''
            Perform dynamic programming on the states in set Z in order to update values and best actions
        '''
        (values, best_actions) = z_value_iteration(mdp, Z, values, best_actions, 0.999, 0.000001)  # *
        return values,best_actions

    # Steps definition  # *
    global expanded, G, Z, best_actions, values, tip
    best_actions = [0] * len(mdp.S)
    values = [0] * len(mdp.S)  # *
    start_state = mdp.S[start_state - 1]
    tip = [False] * len(mdp.S)
    tip[start_state.number - 1] = True
    expanded = None
    G = set([start_state])  # Explicit graph
    Z = set()
    ''' 
        every step function receives the step button object and returns:
        a list of value's lists, a list of labels, a list of best_actions, a list of colors
        every None return will no be updated on GUI
    '''
    def step1(step_button):
        global expanded, G, best_actions, values, tip
        expanded = expand(start_state,best_actions,tip)
        if not expanded:
            step_button['state'] = 'disabled'
        colors = ['#f0f0f0'] * len(mdp.S)
        for s in G:
            if tip[s.number-1]:
                colors[s.number-1] = '#A0F9FF'
            else:
                colors[s.number-1] = '#0A767D'
        if expanded:
            colors[expanded.number-1] = '#FF5151'
        return [values],['V'],best_actions,colors
    def step2(step_button):
        global expanded, G, values, best_actions, tip
        G = successors(expanded,G,tip,values)
        colors = ['#f0f0f0'] * len(mdp.S)
        for s in G:
            if tip[s.number - 1]:
                colors[s.number - 1] = '#A0F9FF'
            else:
                colors[s.number - 1] = '#0A767D'
        if expanded:
            colors[expanded.number - 1] = '#FF5151'
        return [values],['V'],best_actions,colors
    def step3(step_button):
        global expanded, G, Z, best_actions, tip
        Z = setZ(expanded,G,tip,best_actions)
        colors = ['#f0f0f0'] * len(mdp.S)
        for s in G:
            if tip[s.number - 1]:
                colors[s.number - 1] = '#A0F9FF'
            else:
                colors[s.number - 1] = '#0A767D'
        if expanded:
            colors[expanded.number - 1] = '#FF5151'
        for s in Z:
            colors[s.number-1] = '#FFC300'
        return None,None,None,colors
    def step4(step_button):
        global expanded, G, Z, best_actions, values
        (values,best_actions) = dynamic_programming(Z,values,best_actions)
        colors = ['#f0f0f0'] * len(mdp.S)
        for s in G:
            if tip[s.number - 1]:
                colors[s.number - 1] = '#A0F9FF'
            else:
                colors[s.number - 1] = '#0A767D'
        if expanded:
            colors[expanded.number - 1] = '#FF5151'
        for s in Z:
            colors[s.number - 1] = '#FFC300'
        return [values], ['V'], best_actions, colors

    MDP.gui.step_plot(mdp,[step1,step2,step3,step4],['Expansion','Adding successors','Putting ancestor in Z','Dynamic programming'])

# Test Script
# mdp = MDP.MDP(4, 8, 4)
# mdp.set_costs(1)
# MDP.problems.swim_symmetric(mdp.Nx, mdp.Ny, mdp.A, 0.8, 0, True, mdp)
# print(mdp)
#
# (pg_values, risk_values, best_actions) = z_dual_criterion_risk_sensitive(mdp,None,[0 for s in mdp.S],[0 for s in mdp.S],[0 for s in mdp.S])
# MDP.gui.plot(mdp,[pg_values,risk_values],['PG','V'],best_actions)
#
# (cost_values, best_actions) = LAO(mdp,1)
# MDP.gui.plot(mdp, [cost_values], ['V'], best_actions)
#
# LAO_interactive(mdp,1)
