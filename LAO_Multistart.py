import MDP
import LAO


def LAO_Multistart(mdp, start_state=1, processed_set=None, pg_values=None, risk_values=None, best_actions=None):  # *
    """
    The same as the LAO algorithm but this version reuses previous calculations, turning possible to run for another
    start state and to consider the previously calculated values in a way that upon finding a best solution graph,
    if advantageous, it appends the best solution graph and ends the algorithm saving processing

    :param mdp: an MDP object
    :param start_state: an integer representing the start state
    :return: the states in best solution graph, the final values and best actions determined by the dynamic programming

    In order to change the dynamic programming function you have to pay a special attention to:
        - an admissible heuristic values
        - starter values
        - goal values assigned on successors have to be the best values according to the dynamic programming algorithm
        - the dynamic programming algorithm has to keep values and best actions through calls
        - the return of LAO
    All of this changes are marked with a comment '# *' on the code
    """
    if not processed_set or not pg_values or not risk_values or not best_actions:
        processed_set = set()
        pg_values = [0 for s in mdp.S]  # *
        risk_values = [1 for s in mdp.S]  # *
        best_actions = [0 for s in mdp.S]

    # Heuristic
    def h(s):  # *
        return 1

    start_state = mdp.S[start_state - 1]
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
                break
            else:
                for t in s.T[best_actions[s.number - 1]]:
                    s2 = mdp.S[t['state'] - 1]
                    if not visited[s2.number - 1]:
                        bfs.append(s2)
                        visited[s2.number - 1] = True

        # If there are no tips to expand the LAO algorithm is over
        if not expanded or expanded in processed_set:
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
                    elif s2 not in processed_set:  # reusing the previous value of the states in the processed set
                        pg_values[s2.number - 1] = h(s2)

        '''
            Create a set Z that contains the expanded state and all of its ancestors in the explicit graph along
            marked action arcs.
            Here we do from every state in the explicit graph a recursion keeping the path
            Then if the path from a start state x reaches the expanded state we add this path to the set Z
            At the end of this part we will have all the states in the explicit graph that can reach the expanded state
            In other words, all of its ancestors
        '''
        def recursion(s, visited, Z):
            visited[s.number - 1] = True

            found = False

            if s == expanded:
                Z.add(s)  # Z is a set so there is no repetitions
                return True

            for t in s.T[best_actions[s.number - 1]]:
                s2 = mdp.S[t['state'] - 1]
                # if s2 is in G and it is not a tip and it was not visited
                if s2 in G and not tip[s2.number - 1] and not visited[s2.number - 1]:
                    if not found:
                        found = recursion(s2, visited, Z)
                    else:
                        recursion(s2, visited, Z)

            if found:
                Z.add(s)
                return True
            return False

        Z = set()
        for start in G:
            if not tip[start.number - 1]:
                visited = [False for s in mdp.S]
                recursion(start, visited, Z)

        '''
            Perform dynamic programming on the states in set Z in order to update values and best actions
        '''
        (pg_values, risk_values, best_actions) = LAO.z_dual_criterion_risk_sensitive(mdp,Z,pg_values,risk_values,best_actions,-0.01,0.0001)  # *

    '''
        Extracting the best solution graph in order to reuse the calculations in the next calls to this function.
        For that, we call the same recursion function used to build Z but only for the start state
    '''
    bsg = set()
    recursion(start_state, [False for s in mdp.S], bsg)

    '''
        Adding the best solution graph to the processed set and returning it, the values, and best actions
    '''
    processed_set.update(bsg)

    return processed_set, pg_values, risk_values, best_actions  # *


# Test Script
mdp = MDP.MDP(4, 8, 4)
mdp.set_costs(1)
MDP.problems.swim_symmetric(mdp.Nx, mdp.Ny, mdp.A, 0.8, 0, True, mdp)
print(mdp)

# First call
(processed,pg_values,risk_values,best_actions) = LAO_Multistart(mdp,26)
MDP.gui.plot(mdp,[pg_values,risk_values],['PG','V'],best_actions)

# Next calls
(processed,pg_values,risk_values,best_actions) = LAO_Multistart(mdp,1,processed,pg_values,risk_values,best_actions)
MDP.gui.plot(mdp,[pg_values,risk_values],['PG','V'],best_actions)
