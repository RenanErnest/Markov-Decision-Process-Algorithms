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

    return values, best_actions


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


def LAO_interactive(mdp, start_state=1):
    '''
    The same as the LAO function above but divided into functions in order to visualize each step on GUI

    :param mdp: an MDP object
    :param start_state: an integer representing the start state

    In order to change the dynamic programming function you have to pay a special attention to:
        - an admissible heuristic values
        - starter values
        - goal values assigned on successors have to be the best values according to the dynamic programming algorithm used
        - the dynamic programming algorithm has to keep values and best actions through calls
    All of this changes are marked with a comment '# *' on the code
    '''

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
        global expanded, G, values, tip
        G = successors(expanded,G,tip,values)
        colors = ['#f0f0f0'] * len(mdp.S)
        for s in G:
            if tip[s.number - 1]:
                colors[s.number - 1] = '#A0F9FF'
            else:
                colors[s.number - 1] = '#0A767D'
        if expanded:
            colors[expanded.number - 1] = '#FF5151'
        return None,None,None,colors
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
mdp = MDP.MDP(4, 8, 4)
MDP.problems.swim_symmetric(mdp.Nx, mdp.Ny, mdp.A, 0.8, 0, True, mdp)
print(mdp)
mdp.set_costs(1)

(cost_values, best_actions) = LAO(mdp,2)
MDP.gui.plot(mdp, [cost_values], ['V'], best_actions)

LAO_interactive(mdp,1)
