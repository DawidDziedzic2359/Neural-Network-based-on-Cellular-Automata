import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from collections import defaultdict
import os
import re

class Visualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Rule Set Visualizer")
        self.master.geometry("400x100")
        self.figures = []
        self.viz_window = None

        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat")

        self.toolbar_frame = ttk.Frame(master)
        self.toolbar_frame.pack(pady=10)

        self.visualize_btn = ttk.Button(self.toolbar_frame, text="Visualize Rules", command=self.visualize_rule_sets)
        self.visualize_btn.pack(side=tk.LEFT, padx=5)

        self.export_btn = ttk.Button(self.toolbar_frame, text="Export Plots", command=self.export_plots)
        self.export_btn.pack(side=tk.LEFT, padx=5)
        self.export_btn.state(["disabled"])

    def create_scrollable_frame(self, parent):
        container = ttk.Frame(parent)
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        container.pack(fill=tk.BOTH, expand=True)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        return scrollable_frame

    def parse_all_test_sets(self):
        test_set_files = []
        test_set_number = 1
        while os.path.exists(f"test_set_{test_set_number}.txt"):
            test_set_files.append(f"test_set_{test_set_number}.txt")
            test_set_number += 1

        rule_sets = []
        individual_scores = defaultdict(int)

        for file_path in test_set_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    ref_match = re.search(r'Rules from (test_set_\d+)', content)
                    if ref_match:
                        ref_file = ref_match.group(1) + ".txt"
                        with open(ref_file) as ref_f:
                            ref_content = ref_f.read()
                            rules_section = ref_content.split("Rules:\n")[1].split("\n\nRule Scores:")[0]
                    else:
                        rules_section = content.split("Rules:\n")[1].split("\n\nRule Scores:")[0]
                    layers = []
                    for line in rules_section.split('\n'):
                        line = line.strip()
                        if line.startswith('///////////////'): continue
                        try:
                            layers.append(list(map(int, line.split(','))))
                        except (ValueError, IndexError):
                            continue
                    try:
                        scores_section = content.split("\nRule Scores:\n")[1]
                        for line in scores_section.split('\n'):
                            if line.startswith("Rule: "):
                                rule_num = int(re.search(r'Rule: (\d+)', line).group(1))
                                score = int(re.search(r'Score: (-?\d+)', line).group(1))
                                individual_scores[rule_num] += score
                    except IndexError:
                        pass

                    rule_sets.append({
                        'input': content.split("Input: ")[1].split("\n")[0].strip(),
                        'output': content.split("Expected output: ")[1].split("\n")[0].strip(),
                        'rules': layers,
                        'signature': tuple(tuple(layer) for layer in layers)
                    })

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

        return {
            'rule_sets': rule_sets,
            'individual_scores': individual_scores
        }

    def visualize_rule_sets(self):
        if self.viz_window:
            self.viz_window.destroy()

        rule_data = self.parse_all_test_sets()
        self.viz_window = tk.Toplevel(self.master)
        self.viz_window.title("Analysis Results")
        self.viz_window.state('zoomed')

        notebook = ttk.Notebook(self.viz_window)
        notebook.pack(fill=tk.BOTH, expand=True)

        charts_frame = ttk.Frame(notebook)
        self.create_charts_tab(charts_frame, rule_data)
        notebook.add(charts_frame, text="Charts")

        data_frame = ttk.Frame(notebook)
        self.create_data_tab(data_frame, rule_data)
        notebook.add(data_frame, text="Raw Data")

        self.export_btn.state(["!disabled"])

    def plot_individual_rules(self, ax, data):
        sorted_rules = sorted(data['individual_scores'].items(), key=lambda x: x[1], reverse=True)
        filtered_rules = [(k, v) for k, v in sorted_rules if v != 0]

        colors = ['#4caf50' if v > 0 else '#f44336' for _, v in filtered_rules]
        ax.bar([str(r[0]) for r in filtered_rules], [r[1] for r in filtered_rules], color=colors)

        ax.set_title("Individual Rule Performance", fontsize=14)
        ax.set_xlabel("Rule Number", fontsize=12)
        ax.set_ylabel("Total Score", fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.grid(True, alpha=0.3)

    def create_charts(self, parent, data):
        self.figures = []

        fig1 = plt.figure(figsize=(14, 6), dpi=120)
        self.figures.append(fig1)
        ax1 = fig1.add_subplot(111)
        self.plot_individual_rules(ax1, data)
        self.embed_plot(fig1, parent)

        fig2 = plt.figure(figsize=(14, 8), dpi=120)
        self.figures.append(fig2)
        ax2 = fig2.add_subplot(111)
        self.plot_rule_sets(ax2, data)
        self.embed_plot(fig2, parent)

    def plot_rule_sets(self, ax, data):
        set_counts = defaultdict(int)
        for ruleset in data['rule_sets']:
            set_counts[ruleset['signature']] += 1

        sorted_sets = sorted(set_counts.items(), key=lambda x: x[1], reverse=True)
        total_sets = sum(count for _, count in sorted_sets)
        colors = plt.cm.tab20.colors
        color_map = {sig: colors[i % len(colors)] for i, (sig, _) in enumerate(sorted_sets)}

        labels = []
        percentages = []
        for idx, (sig, count) in enumerate(sorted_sets):
            percentage = (count / total_sets) * 100
            layers_str = " | ".join([f"L{li + 1}: {','.join(map(str, layer[:8]))}"
                                     for li, layer in enumerate(sig)])
            labels.append(f"Set {idx + 1} [{percentage:.1f}%]: {layers_str}")
            percentages.append(percentage)

        bars = ax.bar([f"Set {i + 1}" for i in range(len(sorted_sets))], percentages, color=[color_map[s[0]] for s in sorted_sets])

        plt.subplots_adjust(bottom=0.35)

        ax.legend(bars, labels, title="Rule Set Details", loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8, title_fontsize=9, frameon=False)
        ax.set_title("Rule Set Frequency", fontsize=14)
        ax.set_xlabel("Rule Set Group", fontsize=12)
        ax.set_ylabel("Percentage of Total Sets", fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)

    def create_charts_tab(self, parent, data):
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.figures = []

        fig1 = plt.figure(figsize=(16, 6), dpi=120)
        self.figures.append(fig1)
        ax1 = fig1.add_subplot(111)
        self.plot_individual_rules(ax1, data)
        self.embed_plot(fig1, scrollable_frame)

        fig2 = plt.figure(figsize=(14, 8), dpi=120)
        self.figures.append(fig2)
        ax2 = fig2.add_subplot(111)
        self.plot_rule_sets(ax2, data)
        self.embed_plot(fig2, scrollable_frame)

    def embed_plot(self, figure, parent):
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        canvas = FigureCanvasTkAgg(figure, plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        toolbar.pack(side=tk.LEFT)

    def export_plots(self):
        if not self.figures:
            messagebox.showwarning("Warning", "No data to export!")
            return
        file_types = [
            ('PDF', '*.pdf'),
            ('PNG', '*.png'),
            ('SVG', '*.svg')
        ]
        path = filedialog.asksaveasfilename(
            filetypes=file_types,
            defaultextension=".pdf",
            title="Save Visualizations As"
        )

        if not path:
            return
        try:
            if path.lower().endswith('.pdf'):
                with PdfPages(path) as pdf:
                    for fig in self.figures:
                        fig.set_size_inches(14, 8)
                        pdf.savefig(fig, bbox_inches='tight', dpi=300)
            else:
                base = os.path.splitext(path)[0]
                for i, fig in enumerate(self.figures):
                    fig.savefig(f"{base}_{i + 1}.{path.split('.')[-1]}", bbox_inches='tight', dpi=300)
            messagebox.showinfo("Success", f"Visualizations exported to:\n{path}")
            if self.viz_window:
                self.viz_window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")

    def create_data_tab(self, parent, data):
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True)

        text_area = tk.Text(container, wrap=tk.NONE, font=('Consolas', 9))
        vsb = ttk.Scrollbar(container, orient="vertical", command=text_area.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=text_area.xview)
        text_area.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        text_area.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        text_area.insert(tk.END, "Detailed Rule Set Analysis\n\n")
        for i, ruleset in enumerate(data['rule_sets']):
            text_area.insert(tk.END, f"Test Set {i + 1}\n")
            text_area.insert(tk.END, f"Input: {ruleset['input']}\n")
            text_area.insert(tk.END, f"Expected: {ruleset['output']}\n")
            text_area.insert(tk.END, "Rules:\n")
            for li, layer in enumerate(ruleset['rules']):
                text_area.insert(tk.END, f"  Layer {li + 1}: {','.join(map(str, layer))}\n")
            text_area.insert(tk.END, "-" * 80 + "\n")


if __name__ == "__main__":
    root = tk.Tk()
    visualizer = Visualizer(root)
    root.mainloop()