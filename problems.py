'''
    All of the problems here receive as arguments its setups and return a transition matrix
'''

import math

def swim(nx,ny,na,probFlow, probFall, bridge, mdp=None):
    '''
    :param nx: number of states in the x axis
    :param ny: number of states in the y axis
    :param na: number of actions
    :param probFlow: probability to go down the river
    :param probFall: probability to fall in the river when in borders
    :param bridge: if there will be a bridge at the top
    :param mdp: if passed, it will change the transitions inside the mdp object structure
    :return: T: a transition matrix
    '''

    nstates = nx*ny+1 # number of states
    T = [[[0 for s2 in range(nstates)] for s1 in range(nstates)] for a in range(na)]

    for s in range(nx * ny):  # each state
        y = s % ny
        x = math.floor(s / ny)

        if y == 0:  # goal or waterfall
            if x == nx - 1:  # goal
                for a in range(na):
                    # for each action it has 1 of probability to go to a absorb state where costs are zero
                    T[a][s][nstates - 1] = 1
            else:
                for a in range(na):
                    T[a][s][s] = 1;  # for each action it has 1 of probability to go to itself
        else:
            # UP
            a = 0
            if x > 0 and x < nx - 1 and (y < ny - 1 or not bridge):  # rio
                x1 = x;
                x2 = x;
                x3 = x;
                y1 = min(y + 1, ny - 1);
                y2 = y - 1;
                y3 = y;

                s1 = x1 * ny + y1;  # sobe
                s2 = x2 * ny + y2;  # desce
                s3 = x3 * ny + y3;  # parado

                T[a][s][s1] = (1 - probFlow) ** 2;
                T[a][s][s2] = T[a][s][s2] + probFlow ** 2;
                T[a][s][s3] = T[a][s][s3] + 2 * probFlow * (1 - probFlow);
            else:  # margem
                x1 = x;
                y1 = min(y + 1, ny - 1);
                s1 = x1 * ny + y1;
                x2 = min(x + 1, nx - 1);
                y2 = y;
                s2 = x2 * ny + y2;
                T[a][s][s1] = 1 - probFall;
                T[a][s][s2] = T[a][s][s2] + probFall;

            # DOWN
            a = 1
            x1 = x;
            y1 = y - 1;
            s1 = x1 * ny + y1;
            T[a][s][s1] = 1;

            # RIGHT
            a = 2
            if x > 0 and x < nx - 1 and (y < ny - 1 or not bridge):  # rio
                x1 = min(x + 1, nx - 1);
                x2 = x;
                x3 = min(x + 1, nx - 1);
                x4 = x;
                y1 = y;
                y2 = y - 1;
                y3 = y - 1;
                y4 = y;
                s1 = x1 * ny + y1;  # east
                s2 = x2 * ny + y2;  # south
                s3 = x3 * ny + y3;  # southeast
                s4 = x4 * ny + y4;  # stopped

                T[a][s][s1] = (1 - probFlow) ** 2;
                T[a][s][s2] = T[a][s][s2] + probFlow ** 2;
                T[a][s][s3] = T[a][s][s3] + (1 - probFlow) * probFlow;
                T[a][s][s4] = T[a][s][s4] + probFlow * (1 - probFlow);
            else:
                x1 = min(x + 1, nx - 1);
                y1 = y;
                s1 = x1 * ny + y1;
                T[a][s][s1] = 1

            # LEFT
            a = 3
            if x > 0 and x < nx - 1 and (y < ny - 1 or not bridge):  # river
                x1 = max(x - 1, 0);
                x2 = x;
                x3 = max(x - 1, 0);
                x4 = x;
                y1 = y;
                y2 = y - 1;
                y3 = y - 1;
                y4 = y;
                s1 = x1 * ny + y1;  # weast
                s2 = x2 * ny + y2;  # south
                s3 = x3 * ny + y3;  # southwest
                s4 = x4 * ny + y4;  # stopped
                T[a][s][s1] = (1 - probFlow) ** 2;
                T[a][s][s2] = T[a][s][s2] + probFlow ** 2;
                T[a][s][s3] = T[a][s][s3] + (1 - probFlow) * probFlow;
                T[a][s][s4] = T[a][s][s4] + probFlow * (1 - probFlow);
            else:
                x1 = max(x - 1, 0);
                y1 = y;
                s1 = x1 * ny + y1;
                T[a][s][s1] = 1;

    # setting goal
    # goal_state = (nx - 1) * ny
    goal_state = nx * ny
    mdp.S[goal_state].cost = 0
    mdp.S[goal_state].goal = True

    for a in range(na):
        for s2 in range(nstates):
            T[a][(nx - 1) * ny][s2] = 0
        T[a][(nx - 1) * ny][nstates - 1] = 1

    if mdp:
        mdp.matrix_to_edges(T)

    return T


def swim_without_deadend(nx,ny,na,probFlow, probFall, mdp=None):
    '''
    in this implementation the waterfall states have probability 1 to go to the default initial state (2)
    :param nx: number of states in the x axis
    :param ny: number of states in the y axis
    :param na: number of actions
    :param probFlow: probability to go down the river
    :param probFall: probability to fall in the river when in borders
    :param mdp: if passed, it will change the transitions inside the mdp object structure
    :return: T: a transition matrix
    '''

    nstates = nx*ny+1 # number of states
    T = [[[0 for s2 in range(nstates)] for s1 in range(nstates)] for a in range(na)]

    for s in range(nx * ny):  # each state
        y = s % ny
        x = math.floor(s / ny)

        if y == 0 and x != 0:  # goal or waterfall
            if x == nx - 1:  # goal
                for a in range(na):
                    # for each action it has 1 of probability to go to a absorb state where costs are zero
                    T[a][s][nstates - 1] = 1
            else:
                for a in range(na):
                    T[a][s][1] = 1;  # for each action it has 1 of probability to go to the state 2 (initial by default)
        else:
            # UP
            a = 0
            if x > 0 and x < nx - 1 and y < ny - 1:  # rio
                x1 = x;
                x2 = x;
                x3 = x;
                y1 = min(y + 1, ny - 1);
                y2 = y - 1;
                y3 = y;

                s1 = x1 * ny + y1;  # sobe
                s2 = x2 * ny + y2;  # desce
                s3 = x3 * ny + y3;  # parado

                T[a][s][s1] = (1 - probFlow) ** 2;
                T[a][s][s2] = T[a][s][s2] + probFlow ** 2;
                T[a][s][s3] = T[a][s][s3] + 2 * probFlow * (1 - probFlow);
            else:  # margem
                x1 = x;
                y1 = min(y + 1, ny - 1);
                s1 = x1 * ny + y1;
                x2 = min(x + 1, nx - 1);
                y2 = y;
                s2 = x2 * ny + y2;
                T[a][s][s1] = 1 - probFall;
                T[a][s][s2] = T[a][s][s2] + probFall;

            # DOWN
            a = 1
            x1 = x;
            y1 = max(y - 1, 0);
            s1 = x1 * ny + y1;
            T[a][s][s1] = 1;

            # RIGHT
            a = 2
            if x > 0 and x < nx - 1 and y < ny - 1:  # rio
                x1 = min(x + 1, nx - 1);
                x2 = x;
                x3 = min(x + 1, nx - 1);
                x4 = x;
                y1 = y;
                y2 = y - 1;
                y3 = y - 1;
                y4 = y;
                s1 = x1 * ny + y1;  # east
                s2 = x2 * ny + y2;  # south
                s3 = x3 * ny + y3;  # southeast
                s4 = x4 * ny + y4;  # stopped

                T[a][s][s1] = (1 - probFlow) ** 2;
                T[a][s][s2] = T[a][s][s2] + probFlow ** 2;
                T[a][s][s3] = T[a][s][s3] + (1 - probFlow) * probFlow;
                T[a][s][s4] = T[a][s][s4] + probFlow * (1 - probFlow);
            else:
                x1 = min(x + 1, nx - 1);
                y1 = y;
                s1 = x1 * ny + y1;
                T[a][s][s1] = 1

            # LEFT
            a = 3
            if x > 0 and x < nx - 1 and y < ny - 1:  # river
                x1 = max(x - 1, 0);
                x2 = x;
                x3 = max(x - 1, 0);
                x4 = x;
                y1 = y;
                y2 = y - 1;
                y3 = y - 1;
                y4 = y;
                s1 = x1 * ny + y1;  # weast
                s2 = x2 * ny + y2;  # south
                s3 = x3 * ny + y3;  # southwest
                s4 = x4 * ny + y4;  # stopped
                T[a][s][s1] = (1 - probFlow) ** 2;
                T[a][s][s2] = T[a][s][s2] + probFlow ** 2;
                T[a][s][s3] = T[a][s][s3] + (1 - probFlow) * probFlow;
                T[a][s][s4] = T[a][s][s4] + probFlow * (1 - probFlow);
            else:
                x1 = max(x - 1, 0);
                y1 = y;
                s1 = x1 * ny + y1;
                T[a][s][s1] = 1;

    # setting goal
    # goal_state = (nx - 1) * ny
    goal_state = nx * ny
    mdp.S[goal_state].cost = 0
    mdp.S[goal_state].goal = True

    for a in range(na):
        for s2 in range(nstates):
            T[a][(nx - 1) * ny][s2] = 0
        T[a][(nx - 1) * ny][nstates - 1] = 1

    if mdp:
        mdp.matrix_to_edges(T)

    return T


def swim_symmetric(nx,ny,na,probFlow, probFall, bridge, mdp=None):
    '''
    :param nx: number of states in the x axis
    :param ny: number of states in the y axis
    :param na: number of actions
    :param probFlow: probability to go down the river
    :param probFall: probability to fall in the river when in borders
    :param bridge: if there will be a bridge at the top
    :param mdp: if passed, it will change the transitions inside the mdp object structure
    :return: T: a transition matrix
    '''

    nstates = nx*ny+1 # number of states
    T = [[[0 for s2 in range(nstates)] for s1 in range(nstates)] for a in range(na)]

    for s in range(nx * ny):  # each state
        y = s % ny
        x = math.floor(s / ny)

        if y == 0 and x != 0:  # goal or waterfall
            if x == nx - 1:  # goal
                for a in range(na):
                    # for each action it has 1 of probability to go to a absorb state where costs are zero
                    T[a][s][nstates - 1] = 1
            else:
                for a in range(na):
                    T[a][s][s] = 1;  # for each action it has 1 of probability to go to itself
        else:
            # UP
            a = 0
            if x > 0 and x < nx - 1 and (y < ny - 1 or not bridge):  # rio
                x1 = x;
                x2 = x;
                x3 = x;
                y1 = min(y + 1, ny - 1);
                y2 = y - 1;
                y3 = y;

                s1 = x1 * ny + y1;  # sobe
                s2 = x2 * ny + y2;  # desce
                s3 = x3 * ny + y3;  # parado

                T[a][s][s1] = (1 - probFlow) ** 2;
                T[a][s][s2] = T[a][s][s2] + probFlow ** 2;
                T[a][s][s3] = T[a][s][s3] + 2 * probFlow * (1 - probFlow);
            else:  # margem
                x1 = x;
                y1 = min(y + 1, ny - 1);
                s1 = x1 * ny + y1;
                x2 = min(x + 1, nx - 1);
                y2 = y;
                s2 = x2 * ny + y2;
                T[a][s][s1] = 1 - probFall;
                T[a][s][s2] = T[a][s][s2] + probFall;

            # DOWN
            a = 1
            x1 = x;
            #y1 = y - 1;
            y1 = max(y - 1, 0);
            s1 = x1 * ny + y1;
            T[a][s][s1] = 1;

            # RIGHT
            a = 2
            if x > 0 and x < nx - 1 and (y < ny - 1 or not bridge):  # rio
                x1 = min(x + 1, nx - 1);
                x2 = x;
                x3 = min(x + 1, nx - 1);
                x4 = x;
                y1 = y;
                y2 = y - 1;
                y3 = y - 1;
                y4 = y;
                s1 = x1 * ny + y1;  # east
                s2 = x2 * ny + y2;  # south
                s3 = x3 * ny + y3;  # southeast
                s4 = x4 * ny + y4;  # stopped

                T[a][s][s1] = (1 - probFlow) ** 2;
                T[a][s][s2] = T[a][s][s2] + probFlow ** 2;
                T[a][s][s3] = T[a][s][s3] + (1 - probFlow) * probFlow;
                T[a][s][s4] = T[a][s][s4] + probFlow * (1 - probFlow);
            else:
                x1 = min(x + 1, nx - 1);
                y1 = y;
                s1 = x1 * ny + y1;
                T[a][s][s1] = 1

            # LEFT
            a = 3
            if x > 0 and x < nx - 1 and (y < ny - 1 or not bridge):  # river
                x1 = max(x - 1, 0);
                x2 = x;
                x3 = max(x - 1, 0);
                x4 = x;
                y1 = y;
                y2 = y - 1;
                y3 = y - 1;
                y4 = y;
                s1 = x1 * ny + y1;  # weast
                s2 = x2 * ny + y2;  # south
                s3 = x3 * ny + y3;  # southwest
                s4 = x4 * ny + y4;  # stopped
                T[a][s][s1] = (1 - probFlow) ** 2;
                T[a][s][s2] = T[a][s][s2] + probFlow ** 2;
                T[a][s][s3] = T[a][s][s3] + (1 - probFlow) * probFlow;
                T[a][s][s4] = T[a][s][s4] + probFlow * (1 - probFlow);
            else:
                x1 = max(x - 1, 0);
                y1 = y;
                s1 = x1 * ny + y1;
                T[a][s][s1] = 1;

    # setting goal
    # goal_state = (nx - 1) * ny
    goal_state = nx*ny
    mdp.S[goal_state].cost = 0
    mdp.S[goal_state].goal = True

    for a in range(na):
        for s2 in range(nstates):
            T[a][(nx - 1) * ny][s2] = 0
        T[a][(nx - 1) * ny][nstates - 1] = 1

    if mdp:
        mdp.matrix_to_edges(T)

    return T