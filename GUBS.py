import MDP
import LAO
import LAO_Multistart

def LAO_GUBS(mdp, start_state=1):  # *
    """
    In progress...
    We have to think about how to implement the expanded states on demand and include they in the mdp without lose so
    so much in performance

    :param mdp: an MDP object
    :param start_state: an integer representing the start state
    :return: the states in best solution graph, the final values and best actions determined by the dynamic programming
    """

    # Heuristic
    def h(s):  # *
        return 1

    pg_values = [0 for s in mdp.S]  # *
    risk_values = [1 for s in mdp.S]  # *
    best_actions = [0 for s in mdp.S]

    start_state = mdp.S[start_state - 1]
    last_expanded = None
    tip = [False for s in mdp.S]
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
        visited = [False for s in mdp.S]
        while bfs:
            s = bfs.pop(0)
            if tip[s.number - 1]:
                expanded = s
                last_expanded = expanded
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
                        pg_values[s2.number - 1] = 1
                    else:  # reusing the previous value of the states in the processed set
                        pg_values[s2.number - 1] = h(s2)

        '''
            Create a set Z that contains the expanded state and all of its ancestors in the explicit graph along
            marked action arcs.
            Here we do from every state in the explicit graph a recursion keeping the path
            Then if the path from a start state x reaches the expanded state we add this path to the set Z
            At the end of this part we will have all the states in the explicit graph that can reach the expanded state
            In other words, all of its ancestors
        '''

        def recursion(s, visited, ancestors, target):
            visited[s.number - 1] = True

            found = False

            if s == target:
                ancestors.add(s)  # ancestors is a set so there is no repetitions
                return True

            if s not in G or tip[s.number - 1]:
                return False

            for t in s.T[best_actions[s.number - 1]]:
                s2 = mdp.S[t['state'] - 1]
                if not visited[s2.number - 1]:
                    if not found:
                        found = recursion(s2, visited, ancestors, target)
                    else:
                        recursion(s2, visited, ancestors, target)

            if found:
                ancestors.add(s)
                return True
            return False

        Z = set()
        for start in G:
            if not tip[start.number - 1]:
                visited = [False for s in mdp.S]
                recursion(start, visited, Z, expanded)
        '''
            Perform dynamic programming on the states in set Z in order to update values and best actions
        '''
        (pg_values, risk_values, best_actions) = LAO.z_dual_criterion_risk_sensitive(mdp,Z,pg_values,risk_values,best_actions,-0.01,0.0001)  # *

    return pg_values, risk_values, best_actions

# Test Script
mdp = MDP.MDP(4, 8, 4)
mdp.set_costs(1)
MDP.problems.swim_symmetric(mdp.Nx, mdp.Ny, mdp.A, 0.8, 0, True, mdp)
print(mdp)


