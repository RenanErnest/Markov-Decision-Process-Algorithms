from tkinter import *
import tkinter.font as font
from tkinter import PhotoImage

def plot(mdp,functions):
    nx = mdp.Nx
    ny = mdp.Ny
    master = Tk()
    arrows = [PhotoImage(file="images/upArrow.png"), PhotoImage(file="images/downArrow.png"),
              PhotoImage(file="images/rightArrow.png"), PhotoImage(file="images/leftArrow.png")]
    states = []
    n = 1

    for i in range(nx):
        for j in reversed(range(ny)):
            s = str(n) + '\nV:' + str(round(mdp.S[n-1].value,3))
            states.append(Button(master,text=s, image=arrows[mdp.S[n-1].action], compound='center', font=font.Font(size=10)))
            states[n-1].grid(row=j,column=i)
            n+=1

    s = str(n) + '\nV:' + str(round(mdp.S[n-1].value,3))
    states.append(Button(master,text=s, image=arrows[mdp.S[n-1].action], compound='center', font=font.Font(size=10)))
    states[n-1].grid(row=nx-2, column=ny+1)

    func = [0]
    expanded = [None]
    expandeds = set()
    G = [None]
    Z = [None]
    label = Label(master,text="Tips",bg='#A0F9FF')
    label.grid(row=0,column=ny+1)
    label = Label(master, text="Z", bg='#FCFE40')
    label.grid(row=1, column=ny + 1)
    label = Label(master, text="Expandido", bg='#FF5151')
    label.grid(row=0, column=ny + 2)
    label = Label(master, text="G'-Tips", bg='#0A767D')
    label.grid(row=1, column=ny + 2)
    master.title('Etapas aparecerao aqui')

    def step():
        if func[0] == 0:
            master.title('Choose a node to expand')
            expanded[0] = functions[func[0]]()
        elif func[0] == 1:
            master.title('Add sucessors of the expanded to the explicit graph')
            G[0] = functions[func[0]](expanded[0])
        elif func[0] == 2:
            master.title('Z with all ancestors of the expanded node')
            Z[0] = functions[func[0]](expanded[0])
        else:
            master.title('Updating values')
            functions[func[0]](Z[0])
        func[0] = (func[0]+1)%len(functions)

        for n in range(len(states)):
            states[n]['text'] = str(n+1) + '\nV:' + str(round(mdp.S[n].value, 3))
            states[n]['image'] = arrows[mdp.S[n].action]
            states[n]['bg']=None

        if not expanded[0]:
            nextButton['state'] = 'disabled'
        else:
            expandeds.add(expanded[0].number)

        if G[0]:
            for state in G[0]:
                states[state.number-1]['bg'] = '#A0F9FF'
        for n in expandeds:
            states[n - 1]['bg'] = '#0A767D'

        if Z[0] and func[0] == 3 or func[0]==0:
            for state in Z[0]:
                states[state.number-1]['bg'] = '#FCFE40'

        if expanded[0]:
            states[expanded[0].number - 1]['bg'] = '#FF5151'

    nextButton = Button(master,text='Step',command=lambda: step())
    nextButton.grid(row=nx-1,column=ny+1)

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