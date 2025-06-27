import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class NetworkVisualizer:
    def __init__(self, master, input_size, processing_layers, input_state, actual_output, expected_output):
        self.master = master
        self.input_size = input_size
        self.processing_layers = processing_layers
        self.input_state = input_state
        self.actual_output = actual_output
        self.expected_output = expected_output
        self.master.title("Network Structure Visualization")
        self.create_widgets()

    def update_network(self, processing_layers, input_state, actual_output, expected_output):
        self.processing_layers = processing_layers
        self.input_state = input_state
        self.actual_output = actual_output
        self.expected_output = expected_output
        self.draw_network()

    def create_widgets(self):
        self.fig = Figure(figsize=(10, 7), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.draw_network()

    def draw_network(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_title("Network Structure")
        ax.axis('off')

        input_color = '#fff9c4'
        hidden_color = '#e3f2fd'
        correct_color = '#c8e6c9'
        incorrect_color = '#ffcdd2'

        num_processing_layers = len(self.processing_layers)
        layer_sizes = [self.input_size] + [len(layer) for layer in self.processing_layers] + [len(self.expected_output)]
        x_positions = [i * 2 for i in range(len(layer_sizes))]
        layer_labels = ["Input"] + [f"Layer {i+1}" for i in range(num_processing_layers)] + ["Output"]

        for i, (x, size) in enumerate(zip(x_positions, layer_sizes)):
            y_positions = [(j - (size - 1)/2) for j in range(size)]
            y_positions = y_positions[::-1]

            for j, y in enumerate(y_positions):
                if i == 0:
                    color = input_color
                elif i == len(x_positions)-1:
                    expected_bit = self.expected_output[j] if j < len(self.expected_output) else None
                    actual_bit = self.actual_output[j] if j < len(self.actual_output) else None
                    if j < len(self.actual_output) and j < len(self.expected_output):
                        color = correct_color if self.actual_output[j] == self.expected_output[j] else incorrect_color
                    else:
                        color = incorrect_color
                else:
                    color = hidden_color

                circle = plt.Circle((x, y), 0.2, color=color, ec='black')
                ax.add_patch(circle)

                if i == 0:
                    label = str(self.input_state[j]) if j < len(self.input_state) else '?'
                elif i == len(x_positions)-1:
                    label = str(self.actual_output[j]) if j < len(self.actual_output) else '?'
                else:
                    label = str(self.processing_layers[i-1][j]) if j < len(self.processing_layers[i-1]) else '?'

                ax.text(x, y, label, ha='center', va='center', fontsize=8)
            ax.text(x, max(y_positions)+1, layer_labels[i], ha='center', va='bottom')

        for i in range(len(x_positions)-1):
            current_layer = [(x_positions[i], y) for y in
                           [(j - (layer_sizes[i]-1)/2) for j in range(layer_sizes[i])]]
            next_layer = [(x_positions[i+1], y) for y in
                          [(j - (layer_sizes[i+1]-1)/2) for j in range(layer_sizes[i+1])]]

            for (x1, y1) in current_layer:
                for (x2, y2) in next_layer:
                    ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.3)

        legend_elements = [
            Rectangle((0,0), 1, 1, facecolor=input_color, edgecolor='black', label='Input'),
            Rectangle((0,0), 1, 1, facecolor=hidden_color, edgecolor='black', label='Hidden Layers'),
            Rectangle((0,0), 1, 1, facecolor=correct_color, edgecolor='black', label='Correct output bit'),
            Rectangle((0,0), 1, 1, facecolor=incorrect_color, edgecolor='black', label='Incorrect output bit')
        ]
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)

        ax.set_xlim(-1, x_positions[-1]+1)
        ymin = -(max(layer_sizes) // 2) - 1
        ymax = max(layer_sizes) // 2 + 1
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        self.canvas.draw()


        