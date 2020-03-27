

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
                    elif state not in processed: # resusing the previous value of the states in the processed set
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
        Extracting the expanded graph obtained by the LAOStar algorithm.
        We keep this graph in order to reuse it if another call to this algorithm ends up
        trying to calculate a node that was already calculated by previous calls.
    '''
    return G