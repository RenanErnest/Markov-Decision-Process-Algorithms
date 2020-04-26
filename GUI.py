from tkinter import *
import tkinter.font as font
from tkinter import PhotoImage


def init_grid():
    root = Tk()
    frame = Frame(root)
    Grid.rowconfigure(root, 0, weight=1)
    Grid.columnconfigure(root, 0, weight=1)
    frame.grid(row=0, column=0, sticky=N + S + E + W)
    grid = Frame(frame)
    grid.grid(sticky=N + S + E + W, column=0, row=7, columnspan=2)
    Grid.rowconfigure(frame, 7, weight=1)
    Grid.columnconfigure(frame, 0, weight=1)
    return root,frame


def plot(mdp, values, labels, best_actions):
    """
    :param mdp: a mdp object
    :param values: a list of all the values for each state that will be plotted like risk value, prob to reach the goal,
        etc, so this param is commonly a list of lists
    :param labels: a list of labels to be plotted with the values, for example 'rv','pg',etc
    """
    (root,frame) = init_grid()
    nx = mdp.Nx
    ny = mdp.Ny
    arrows = [PhotoImage(file="images/upArrow.png"), PhotoImage(file="images/downArrow.png"),
              PhotoImage(file="images/rightArrow.png"), PhotoImage(file="images/leftArrow.png")]
    states = []

    n = 1
    for i in range(nx):
        for j in reversed(range(ny)):

            s = str(n)
            for k in range(len(values)):
                s += '\n' + labels[k] + ':' + str(round(values[k][n - 1], 3))

            states.append(
                Button(frame, text=s, image=arrows[best_actions[n - 1]], compound='center', font=font.Font(size=10)))
            states[n - 1].grid(row=j, column=i, sticky=N + S + E + W)
            n += 1

    s = str(n)
    for k in range(len(values)):
        s += '\n' + labels[k] + ':' + str(round(values[k][n - 1], 3))
    states.append(Button(frame, text=s, image=arrows[best_actions[n - 1]], compound='center', font=font.Font(size=10)))
    states[n - 1].grid(row=ny - 1, column=nx + 3, sticky=N + S + E + W)

    for i in range(nx):
        Grid.columnconfigure(frame, i, weight=1)

    for i in range(ny + 1):
        Grid.rowconfigure(frame, i, weight=1)

    mainloop()


f = 0  # global variable used on step_plot function

def step_plot(mdp,stepFunctions,titles=None):
    '''
    :param mdp: a mdp object
    :param stepFunctions: an in order list of step functions
    :param titles: a list of titles for each step
    '''

    global f
    f = 0

    def step():
        global f
        # getting values
        (values, labels, best_actions, colors) = stepFunctions[f](stepButton)


        # updating values
        if values:
            n = 1
            for state in states:
                s = str(n)
                for k in range(len(values)):
                    s += '\n' + labels[k] + ':' + str(round(values[k][n - 1], 3))
                state['text'] = s
                state['image'] = arrows[best_actions[n-1]]
                n += 1

        # updating title
        if titles:
            root.title(titles[f])

        # updating colors
        if colors:
            for i in range(len(states)):
                states[i]['bg'] = colors[i]

        # increment function index
        f = (f + 1) % len(stepFunctions)

    (root,frame) = init_grid()
    nx = mdp.Nx
    ny = mdp.Ny
    arrows = [PhotoImage(file="images/upArrow.png"), PhotoImage(file="images/downArrow.png"),
              PhotoImage(file="images/rightArrow.png"), PhotoImage(file="images/leftArrow.png")]
    states = []

    n = 1
    for i in range(nx):
        for j in reversed(range(ny)):
            s = str(n)
            states.append(
                Button(frame, text=s, image=arrows[0], compound='center', font=font.Font(size=10)))
            states[n - 1].grid(row=j, column=i, sticky=N + S + E + W)
            n += 1

    s = str(n)
    states.append(Button(frame, text=s, image=arrows[0], compound='center', font=font.Font(size=10)))
    states[n - 1].grid(row=ny - 2, column=nx + 3, sticky=N + S + E + W)

    for i in range(nx):
        Grid.columnconfigure(frame, i, weight=1)

    for i in range(ny + 1):
        Grid.rowconfigure(frame, i, weight=1)

    stepButton = Button(frame, text='Step', compound='right', command=lambda: step())
    stepButton.grid(row=ny-1, column=nx + 3, sticky=N + S + E + W)

    mainloop()