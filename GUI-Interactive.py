from tkinter import *
import tkinter.font as font
from tkinter import PhotoImage


def plot(mdp, values, labels, best_actions):
    """
    :param mdp: a mdp object
    :param values: a list of all the values for each state that will be plotted like risk value, prob to reach the goal,
        etc, so this param is commonly a list of lists
    :param labels: a list of labels to be plotted with the values, for example 'rv','pg',etc
    """
    nx = mdp.Nx
    ny = mdp.Ny
    root = Tk()
    frame = Frame(root)
    Grid.rowconfigure(root, 0, weight=1)
    Grid.columnconfigure(root, 0, weight=1)
    frame.grid(row=0, column=0, sticky=N + S + E + W)
    grid = Frame(frame)
    grid.grid(sticky=N + S + E + W, column=0, row=7, columnspan=2)
    Grid.rowconfigure(frame, 7, weight=1)
    Grid.columnconfigure(frame, 0, weight=1)

    arrows = [PhotoImage(file="images/upArrow.png"), PhotoImage(file="images/downArrow.png"),
              PhotoImage(file="images/rightArrow.png"), PhotoImage(file="images/leftArrow.png")]
    states = []

    n = 1
    for i in range(nx):
        for j in reversed(range(ny)):

            s = str(n) + '\nc:' + str(round(mdp.S[n - 1].cost, 3))
            for k in range(len(values)):
                s += '\n'+labels[k]+':' + str(round(values[k][n-1]))

            states.append(Button(frame, text=s, image=arrows[best_actions[n-1]], compound='center', font=font.Font(size=10)))
            states[n - 1].grid(row=j, column=i, sticky=N + S + E + W)
            n += 1

    s = str(n) + '\nc:' + str(round(mdp.S[n - 1].cost, 3))
    for k in range(len(values)):
        s += '\n' + labels[k] + ':' + str(round(values[k][n - 1]))
    states.append(Button(frame, text=s, image=arrows[best_actions[n - 1]], compound='center', font=font.Font(size=10)))
    states[n - 1].grid(row=ny-2, column=nx+3, sticky=N + S + E + W)

    for i in range(nx):
        Grid.columnconfigure(frame, i, weight=1)

    for i in range(ny + 1):
        Grid.rowconfigure(frame, i, weight=1)

    # root.title('Etapas aparecerao aqui')


    nextButton = Button(frame, text='Step', compound='right', command=lambda: step())
    nextButton.grid(row=ny - 1, column=nx + 3, sticky=N + S + E + W)

    runButton = Button(frame, text='Run', compound='right', command=lambda: run())
    runButton.grid(row=ny, column=nx + 3, sticky=N + S + E + W)

    mainloop()
