from tkinter import *
import tkinter.font as font
from tkinter import PhotoImage

def plot(mdp,functions):
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
            s = str(n) + '\nV:' + str(round(mdp.S[n-1].value,3)) + '\nPG:' + str(round(mdp.S[n-1].probReachGoal, 3))
            states.append(Button(frame,text=s, image=arrows[mdp.S[n-1].action], compound='center', font=font.Font(size=10)))
            states[n-1].grid(row=j,column=i,sticky=N+S+E+W)
            n+=1

    s = str(n) + '\nV:' + str(round(mdp.S[n-1].value,3)) + '\nPG:' + str(round(mdp.S[n-1].probReachGoal, 3))
    states.append(Button(frame,text=s, image=arrows[mdp.S[n-1].action], compound='center', font=font.Font(size=10)))
    states[n-1].grid(row=ny-2, column=nx+3,sticky=N+S+E+W)

    for i in range(nx):
        Grid.columnconfigure(frame, i, weight=1)

    for i in range(ny+1):
        Grid.rowconfigure(frame, i, weight=1)

    func = [0]
    expanded = [None]
    expandeds = set()
    G = [set()] # Explicit graph
    Z = [None]
    startState=[True]
    label = Label(frame,text="Tips",bg='#A0F9FF')
    label.grid(row=0,column=nx+2)
    label = Label(frame, text="Z", bg='#FFC300')
    label.grid(row=1, column=nx + 2)
    label = Label(frame, text="Expandido", bg='#FF5151')
    label.grid(row=0, column=nx + 3)
    label = Label(frame, text="G'-Tips", bg='#0A767D')
    label.grid(row=1, column=nx + 3)
    root.title('Etapas aparecerao aqui')

    def run():
        step()
        while expanded[0]:
            step()
        runButton['state'] = 'disabled'

    def step():
        if func[0] == 0:
            root.title('Choose a node to expand')
            expanded[0] = functions[func[0]]()
            for state in states:
                state['bg'] = '#f0f0f0' #default color

            if G[0]:
                for state in G[0]:
                    states[state.number-1]['bg'] = '#A0F9FF'

            for n in expandeds:
                states[n - 1]['bg'] = '#0A767D'

            if expanded[0]:
                states[expanded[0].number - 1]['bg'] = '#FF5151'

            if startState[0]:
                states[expanded[0].number-1]['fg']='white'
                G[0].add(expanded[0])
                startState[0]=False

        elif func[0] == 1:
            root.title('Add sucessors of the expanded to the explicit graph')
            G[0] = functions[func[0]](expanded[0],G[0])

            if G[0]:
                for state in G[0]:
                    states[state.number - 1]['bg'] = '#A0F9FF'
            for n in expandeds:
                states[n - 1]['bg'] = '#0A767D'

            if expanded[0]:
                states[expanded[0].number - 1]['bg'] = '#FF5151'

        elif func[0] == 2:
            root.title('Z with all ancestors of the expanded node')
            Z[0] = functions[func[0]](expanded[0],G[0])

            if Z[0]:
                for state in Z[0]:
                    states[state.number - 1]['bg'] = '#FFC300'

            # if expanded[0]:
            #     states[expanded[0].number - 1]['bg'] = '#FF5151'
        else:
            root.title('Updating values')
            functions[func[0]](Z[0])
        func[0] = (func[0]+1)%len(functions)

        for n in range(len(states)):
            states[n]['text'] = str(n+1) + '\nV:' + str(round(mdp.S[n].value, 3)) + '\nPG:' + str(round(mdp.S[n].probReachGoal, 3))
            states[n]['image'] = arrows[mdp.S[n].action]
            states[n]['bg']=None

        if not expanded[0]:
            nextButton['state'] = 'disabled'
        else:
            expandeds.add(expanded[0].number)

    nextButton = Button(frame,text='Step',compound='right',command=lambda: step())
    nextButton.grid(row=ny-1,column=nx+3,sticky=N+S+E+W)

    runButton = Button(frame, text='Run', compound='right', command=lambda: run())
    runButton.grid(row=ny, column=nx + 3,sticky=N+S+E+W)

    mainloop()


'''
button arguments:
activebackground: to set the background color when button is under the cursor.
activeforeground: to set the foreground color when button is under the cursor.
bg: to set he normal background color.
command: to call a function.
font: to set the font on the button label.
image: to set the image on the button.
width: to set the width of the button.
height: to set the height of the button.
'''