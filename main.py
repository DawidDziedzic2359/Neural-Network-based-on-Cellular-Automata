from automata.simulation import Simulation
from gui.main_gui import GUI
import tkinter as tk

def main():
    simulation = Simulation(n=101)
    root = tk.Tk()
    gui = GUI(root, simulation)
    root.mainloop()

if __name__ == "__main__":
    main()