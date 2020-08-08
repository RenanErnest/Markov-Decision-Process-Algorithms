import math
import MDP
import LAO
from LAO_Multistart import LAO_Multistart


class Extended_State:
    def __init__(self, number: int, cost: int, goal: bool, transition_matrix, accumulated_cost):
        self.number = number  # label of this state
        self.cost = cost  # static cost taken from following any action
        self.goal = goal  # goal flag
        self.T = transition_matrix
        self.accumulated_cost = accumulated_cost


def LAO_GUBS(mdp, start_state=1, kg=1, risk_factor= -0.01):  # *
    """
    In progress...
    We have to think about how to implement the expanded states on demand and include they in the mdp without lose so
    so much in performance

    :param mdp: an MDP object
    :param start_state: an integer representing the start state
    :param kg: a goal terminal reward constant
    :return: the states in best solution graph, the final values and best actions determined by the dynamic programming
    """

    # Heuristic
    def h(s):  # *
        return 1

    def check_optimal_stationary_policy():
        pass

    # LAO_Multistart variables
    ms_processed = None
    ms_pg_values = None
    ms_risk_values = None
    ms_best_actions = None

    extended_states = {(s, 0): Extended_State(s.number, s.cost, s.goal, s.T, 0) for s in mdp.S}
    pg_values = {s: 0 for s in extended_states.values()}
    risk_values = {s: 1 for s in extended_states.values()}
    best_actions = {s: 0 for s in extended_states.values()}
    tip = {s: False for s in extended_states.values()}
    Cmax = {}

    start_state = extended_states[(mdp.S[start_state - 1], 0)]
    tip[start_state] = True
    G = set([start_state])  # Explicit graph

    while True:

        '''
            Expand some nonterminal tip state n of the best partial solution graph
            We do a breadth-first search from the start state following the best actions until reach a tip
        '''
        # BFS
        expanded = None

        bfs = [start_state]
        visited = {s: False for s in extended_states.values()}
        while bfs:
            s = bfs.pop(0)
            if tip[s]:
                expanded = s
                break
            else:
                for t in s.T[best_actions[s]]:
                    s2 = mdp.S[t['state'] - 1]

                    ac = s.accumulated_cost + s.cost  # accumulated cost after the action

                    # if (s2, ac) not in extended_states:  # if this is true, we need to create the extended state
                    #     s2e = Extended_State(s2.number, s2.cost, s2.goal, s2.T, ac)
                    #     extended_states[(s2, ac)] = s2e
                    #     pg_values[s2e] = 0
                    #     risk_values[s2e] = 1
                    #     best_actions[s2e] = 0
                    #     tip[s2e] = False
                    #     visited[s2e] = False

                    s2e = extended_states[(s2, ac)]

                    if not visited[s2e]:
                        bfs.append(s2e)
                        visited[s2e] = True

        # If there are no tips to expand the LAO algorithm is over
        if not expanded:
            break

        # The expanded node chosen is no more a tip
        tip[expanded] = False

        # Check Optimal stationary policy
        (ms_processed, ms_pg_values, ms_risk_values, ms_best_actions) = LAO_Multistart(mdp, expanded.number,
                                                                                       ms_processed, ms_pg_values,
                                                                                       ms_risk_values,
                                                                                       ms_best_actions)

        '''test stationary optimal policy'''
        if expanded not in Cmax.keys():
            Xset = set()
            for a in range(mdp.A):
                summ = 0
                for t in expanded.T[a]:
                    s2 = mdp.S[t['state'] - 1]
                    (ms_processed, ms_pg_values, ms_risk_values, ms_best_actions) = LAO_Multistart(mdp, s2.number,
                                                                                                   ms_processed,
                                                                                                   ms_pg_values,
                                                                                                   ms_risk_values,
                                                                                                   ms_best_actions)
                    summ += t['prob'] * math.exp(risk_factor * expanded.cost) * ms_risk_values[s2.number - 1]
                if summ > ms_risk_values[expanded.number - 1]:
                    Xset.add(a)
            Wmax = -float('Inf')
            for a in Xset:
                summ = 0
                summ2 = 0
                for t in expanded.T[a]:
                    s2 = mdp.S[t['state'] - 1]
                    summ += t['prob'] * math.exp(risk_factor * expanded.cost) * ms_risk_values[s2.number - 1]
                    summ2 += t['prob'] * ms_pg_values[s2.number - 1] - ms_pg_values[expanded.number - 1]
                W = -1 / risk_factor * math.log((ms_risk_values[expanded.number - 1] - summ) / kg(summ2))
                if W > Wmax:
                    Wmax = W
            Cmax[expanded] = Wmax

        '''add any new successor states to G following any action from the expanded state.'''
        if expanded.accumulated_cost < Cmax[expanded]:  # if it does not pass the test, expand
            for a in range(mdp.A):
                ac = s.accumulated_cost + s.cost  # accumulated cost of each successor state
                for t in expanded.T[a]:
                    s2 = mdp.S[t['state'] - 1]

                    if (s2, ac) not in extended_states:  # if this is true, we need to create the extended state
                        s2e = Extended_State(s2.number, s2.cost, s2.goal, s2.T, ac)
                        extended_states[(s2, ac)] = s2e
                        pg_values[s2e] = 0
                        risk_values[s2e] = 1
                        best_actions[s2e] = 0
                        tip[s2e] = False
                        visited[s2e] = False

                    s2e = extended_states[(s2, ac)]
                    if s2e not in G:
                        G.add(s2e)
                        tip[s2e] = True
                        if s2e.goal:  # *
                            # this value has to be the best value according the dynamic programming algorithm used
                            pg_values[s2e] = 1
                        else:  # reusing the previous value of the states in the processed set
                            pg_values[s2e] = h(s2e)

        '''
            Create a set Z that contains the expanded state and all of its ancestors in the explicit graph along
            marked action arcs.
            Here we do from every state in the explicit graph a recursion keeping the path
            Then if the path from a start state x reaches the expanded state we add this path to the set Z
            At the end of this part we will have all the states in the explicit graph that can reach the expanded state
            In other words, all of its ancestors
        '''
        def recursion(s, visited, ancestors, target):
            visited[s] = True

            found = False

            if s == target:
                ancestors.add(s)  # ancestors is a set so there is no repetitions
                return True

            if s not in G or tip[s]:
                return False

            for t in s.T[best_actions[s]]:
                s2 = mdp.S[t['state'] - 1]
                ac = s.accumulated_cost + s.cost
                s2e = extended_states[(s2, ac)]
                if not visited[s2e]:
                    if not found:
                        found = recursion(s2e, visited, ancestors, target)
                    else:
                        recursion(s2e, visited, ancestors, target)

            if found:
                ancestors.add(s)
                return True
            return False

        Z = set()
        for start in G:
            if not tip[start]:
                visited = {s: False for s in extended_states.values()}
                recursion(start, visited, Z, expanded)

        '''
            Perform dynamic programming on the states in set Z in order to update values and best actions
        '''
        (pg_values, risk_values, best_actions) = GUBS(start_state, extended_states, Z, Cmax, pg_values, risk_values, best_actions,
                                                      ms_pg_values, ms_risk_values, ms_best_actions, risk_factor, kg)  # *

    return pg_values, risk_values, best_actions

def GUBS(start_state, extended_states, Z, Cmax, pg_values, risk_values, best_actions, ms_pg_values, ms_risk_values, ms_best_actions,
             risk_factor=-0.01, goal_reward=1):

    def recursion(s, visited, pg_values, risk_values, best_actions):
        visited[s] = True

        if s not in Z:
            return {'p': pg_values[s], 'v': risk_values[s]}

        if Cmax[s] <= s.accumulated_cost:
            pg_values[s] = ms_pg_values[s.number -1]
            risk_values[s] = ms_risk_values[s.number -1]
            best_actions[s] = ms_best_actions[s.number -1]
            return {'p': pg_values[s], 'v': risk_values[s]}

        maxi = {'total': -float('Inf'), 'a': 0, 'p': None, 'v': None}
        for a in range(mdp.A):
            summV = 0
            summP = 0
            ac = s.accumulated_cost + s.cost
            for t in s.T[a]:
                s2 = mdp.S[t['state'] - 1]
                s2e = extended_states[(s2, ac)]
                if visited[s2e]:
                    summV += t['prob'] * risk_values[s2e]
                    summP += t['prob'] * pg_values[s2e]
                else:
                    s2eMaxi = recursion(s2e, visited, pg_values, risk_values, best_actions)
                    summV += t['prob'] * s2eMaxi['v']
                    summP += t['prob'] * s2eMaxi['p']

            value = math.exp(risk_factor * s.cost) * summV
            probability = summP
            total = math.exp(risk_factor * s.accumulated_cost) * value + goal_reward * probability
            if total > maxi['total']:
                maxi = {'total': total, 'a': a, 'p': probability, 'v': value}

        pg_values[s] = maxi['p']
        risk_values[s] = maxi['v']
        best_actions[s] = maxi['a']

        return maxi

    visited = {s: False for s in extended_states.values()}
    recursion(start_state, visited, pg_values, risk_values, best_actions)
    return pg_values, risk_values, best_actions


# Test Script
mdp = MDP.MDP(4, 8, 4)
mdp.set_costs(1)
MDP.problems.swim_symmetric(mdp.Nx, mdp.Ny, mdp.A, 0.8, 0, True, mdp)
print(mdp)
LAO_GUBS(mdp)
