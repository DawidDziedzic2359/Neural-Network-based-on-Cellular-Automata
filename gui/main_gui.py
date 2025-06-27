import os
import re
import tkinter as tk
from collections import defaultdict
from tkinter import messagebox, filedialog, ttk
import random
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from automata.cellular_automaton import CellularAutomaton
from automata.simulation import Simulation

class GUI:
    def __init__(self, master, simulation):
        self.master = master
        self.simulation = simulation
        master.title("Cellular Automaton Simulator")
        self.visualize_check_state_before_training = False
        self.successful_run = False
        self.temp_initial_state = None
        self.notebook = ttk.Notebook(master)
        self.main_tab = ttk.Frame(self.notebook)
        self.training_tab = ttk.Frame(self.notebook)
        self.dynamic_iteration_count = 0
        self.notebook.add(self.main_tab, text="Main")
        self.notebook.add(self.training_tab, text="Network Training Results")
        self.notebook.pack(expand=1, fill="both")
        self.min_percentage_threshold = 0.00
        self.setup_main_tab()
        self.setup_training_tab()

        self.rules_layers = []
        self.initial_state = None
        self.is_running_test_set = False
        self.visualization_window = None
        self.visualization_canvas = None
        self.missing_test_set = 1
        self.simulation = Simulation(gui=self)

    def setup_main_tab(self):
        # Initial State
        self.initial_state_label = tk.Label(self.main_tab, text="Initial State:")
        self.initial_state_label.grid(row=0, column=0, sticky="w")
        self.initial_state_entry = tk.Entry(self.main_tab, validate="key")
        self.initial_state_entry['validatecommand'] = (
        self.initial_state_entry.register(self.validate_initial_state), '%P')
        self.initial_state_entry.grid(row=0, column=1, sticky="w")

        # Random - Number of sequences - label
        self.random_num_entry_label = tk.Label(self.main_tab, text="Number Of Sequences")
        self.random_num_entry_label.grid(row=0, column=3, sticky="w")
        self.random_num_entry_label.grid_remove()

        self.initial_state_method = tk.StringVar(value="manual")
        self.manual_initial_button = tk.Radiobutton(self.main_tab, text="Manual", variable=self.initial_state_method, value="manual", command=self.update_initial_state_input)
        self.from_file_initial_button = tk.Radiobutton(self.main_tab, text="From File", variable=self.initial_state_method, value="file", command=lambda: (self.update_initial_state_input(), self.load_initial_state_from_file()))
        self.random_initial_button = tk.Radiobutton(self.main_tab, text="Random", variable=self.initial_state_method, value="random", command=self.update_initial_state_input)

        self.manual_initial_button.grid(row=1, column=0, sticky="w")
        self.from_file_initial_button.grid(row=1, column=1, sticky="w")
        self.random_initial_button.grid(row=1, column=2, sticky="w")

        # Random - Number of sequences
        self.random_num_entry = tk.Entry(self.main_tab)
        self.random_num_entry.grid(row=1, column=3, sticky="w")
        self.random_num_entry.config(validate="key", state="disabled", validatecommand=(self.random_num_entry.register(self.validate_random_num), '%P'))
        self.random_num_entry.insert(0, "1")
        self.random_num_entry.grid_remove()

        # Hidden Layers
        self.hidden_layers_label = tk.Label(self.main_tab, text="Hidden Layers:")
        self.hidden_layers_label.grid(row=2, column=0, sticky="w")
        self.hidden_layers_entry = tk.Entry(self.main_tab, validate="key")
        self.hidden_layers_entry['validatecommand'] = (self.hidden_layers_entry.register(self.validate_layers), '%P')
        self.hidden_layers_entry.grid(row=2, column=1, sticky="w")
        self.hidden_layers_entry.insert(0, "0")

        # Size of Each Sequence
        self.size_label = tk.Label(self.main_tab, text="Size of Each Sequence:")
        self.size_label.grid(row=3, column=0, sticky="w")
        self.size_entry = tk.Entry(self.main_tab, validate="key")
        self.size_entry['validatecommand'] = (self.size_entry.register(self.validate_size), '%P')
        self.size_entry.grid(row=3, column=1, sticky="w")

        # Expected Result
        self.expected_result_label = tk.Label(self.main_tab, text="Expected result:")
        self.expected_result_label.grid(row=4, column=0, sticky="w")
        self.expected_result_entry = tk.Entry(self.main_tab, validate="key")
        self.expected_result_entry['validatecommand'] = (
        self.expected_result_entry.register(self.validate_expected_result), '%P')
        self.expected_result_entry.grid(row=4, column=1, sticky="w")

        # Rules
        self.rules_label = tk.Label(self.main_tab, text="Rules:")
        self.rules_label.grid(row=5, column=0, sticky="w")
        self.rules_input_method = tk.StringVar(value="random")
        self.manual_button = tk.Radiobutton(self.main_tab, text="Manual", variable=self.rules_input_method, value="manual", command=self.update_rules_input)
        self.file_button = tk.Radiobutton(self.main_tab, text="From File", variable=self.rules_input_method, value="file", command=self.update_rules_input)
        self.random_button = tk.Radiobutton(self.main_tab, text="Random", variable=self.rules_input_method, value="random", command=self.update_rules_input)
        self.manual_button.grid(row=5, column=1, sticky="w")
        self.file_button.grid(row=5, column=2, sticky="w")
        self.random_button.grid(row=5, column=3, sticky="w")
        self.dynamic_input_frame = tk.Frame(self.main_tab)
        self.dynamic_input_frame.grid(row=6, column=0, columnspan=4, sticky="w")

        self.initial_state_dynamic_frame = tk.Frame(self.main_tab)
        self.initial_state_dynamic_frame.grid(row=0, column=1, sticky="w")

        # Crossover Label and Checkbox
        self.crossover_frame = tk.Frame(self.main_tab)
        self.crossover_frame.grid(row=7, column=0, columnspan=4, sticky="w", pady=5)
        self.crossover_check_var = tk.BooleanVar(value=False)
        self.crossover_check = tk.Checkbutton(
            self.crossover_frame,
            text="Custom Crossover Probability (%):",
            variable=self.crossover_check_var,
            command=self.toggle_crossover_entry
        )
        self.crossover_check.grid(row=0, column=0, sticky="w")
        self.crossover_percent_var = tk.StringVar(value="20")
        self.crossover_percent_entry = tk.Entry(
            self.crossover_frame,
            width=5,
            textvariable=self.crossover_percent_var,
            state='disabled',
            validate="key"
        )
        self.crossover_percent_entry.grid(row=0, column=1, sticky="w", padx=5)
        self.crossover_percent_entry['validatecommand'] = (
            self.crossover_percent_entry.register(self.validate_crossover_percent),
            '%P'
        )
        # Crossover Iterations Checkbox
        self.dynamic_adjust_var = tk.BooleanVar(value=False)
        self.dynamic_adjust_check = tk.Checkbutton(
            self.crossover_frame,
            text="Dynamic Adjustment (-0.1% per iteration)",
            variable=self.dynamic_adjust_var
        )
        self.dynamic_adjust_check.grid(row=0, column=2, padx=5, sticky="w")

        # Dynamic Adjustment Checkbox
        self.dynamic_adjust_var = tk.BooleanVar(value=False)
        self.dynamic_adjust_check = tk.Checkbutton(
            self.crossover_frame,
            text="Dynamic Adjustment (-0.01% per iteration)",
            variable=self.dynamic_adjust_var,
            command=self.toggle_dynamic_adjustment
        )
        self.dynamic_adjust_check.grid(row=0, column=2, padx=5, sticky="w")

        # Additional window for Dynamic Adjustment
        self.dynamic_adjustment_window = None
        self.dynamic_adjustment_label = None
        self.dynamic_adjustment_entry = None

        # Step By Step
        self.step_button = tk.Button(self.main_tab, text="Step By Step", command=self.step_by_step_action)
        self.step_button.grid(row=8, column=0, columnspan=4)

        # Visualize Network Checkbox
        self.visualize_network_var = tk.BooleanVar()
        self.visualize_check = tk.Checkbutton(self.main_tab, text="Visualize Network", variable=self.visualize_network_var)
        self.visualize_check.grid(row=8, column=0, sticky="w")

        # Train Until Success
        self.train_until_success_button = tk.Button(self.main_tab, text="Train Until Success", command=self.train_until_success)
        self.train_until_success_button.grid(row=9, column=0, columnspan=4)

        # Initial State History
        self.history_label = tk.Label(self.main_tab, text="Intermediate States History:")
        self.history_label.grid(row=10, column=0, sticky="w")
        self.history_text = tk.Text(self.main_tab, width=65, height=6, state="disabled")
        self.history_text.grid(row=11, column=0, columnspan=4, sticky="w")

        # Results
        self.results_label = tk.Label(self.main_tab, text="Results:")
        self.results_label.grid(row=12, column=0, sticky="w")
        self.results_text = tk.Text(self.main_tab, width=65, height=6, state="disabled")
        self.results_text.grid(row=13, column=0, columnspan=4, sticky="w")

        # Frame for elements below the label
        self.results_frame = tk.Frame(self.main_tab)
        self.results_frame.grid(row=14, column=0, columnspan=4, sticky="w")

        # Number of Iterations
        self.iterations_label = tk.Label(self.results_frame, text="Number of Iterations: (1-20)")
        self.iterations_label.grid(row=0, column=0, sticky="w")
        self.iterations_label_entry = tk.Entry(self.results_frame, validate="key", state="disabled")
        self.iterations_label_entry['validatecommand'] = (
        self.iterations_label_entry.register(self.validate_numeric), '%P')
        self.iterations_label_entry.grid(row=0, column=1, sticky="w")

        # Run Test_set Files
        self.test_set_files_button = tk.Button(self.results_frame, text="Run 'test_set' Files", command=self.run_test_set_files, state="disabled")
        self.test_set_files_button.grid(row=0, column=2, padx=(5, 0))

        # Label Check Network Training Results
        self.check_network_training_results_label = tk.Label(self.main_tab, text="Network Training Results", font=("Arial", 10))
        self.check_network_training_results_label.grid(row=15, column=0, columnspan=4, pady=10)
        self.update_rules_input()

        # Visualize
        self.visualize_button = tk.Button(self.main_tab, text="Visualize Scores", command=self.open_visualizer, state="disabled")
        self.visualize_button.grid(row=16, column=0, columnspan=4)

    def step_by_step_action(self):
        if not hasattr(self, 'first_step_click'):
            self.first_step_click = True
            self.temp_initial_state = self.initial_state_entry.get()
        if self.dynamic_adjust_var.get() and self.dynamic_iteration_count < 1:
            self.dynamic_iteration_count = 0
        self.is_step_by_step = True
        self.visualize_check_state_before_training = self.visualize_network_var.get()
        self.run_simulation()
        if not self.simulation.successful_run:
            self.train_network()

    def open_visualizer(self):
        import visualization.visualizer
        root = tk.Toplevel(self.master)
        visualizer = visualization.visualizer.Visualizer(root)
        root.geometry("300x150")
        self.center_window(root)

    def center_window(self, window):
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        x = (self.master.winfo_width() // 2) - (width // 2) + self.master.winfo_x()
        y = (self.master.winfo_height() // 2) - (height // 2) + self.master.winfo_y()
        window.geometry(f"{width}x{height}+{x}+{y}")

    def visualize_current_network(self):
        try:
            input_size = int(self.size_entry.get())
            processing_layers = self.simulation.rules_layers

            initial_state_str = self.initial_state_entry.get().strip()
            if not initial_state_str:
                return
            initial_state = [int(bit) for bit in initial_state_str.split(',')[0].strip()]
            expected_output = [int(bit) for bit in self.expected_result_entry.get().strip()]

            current_state = initial_state.copy()
            for layer in processing_layers:
                new_state = []
                for rule in layer:
                    automaton = CellularAutomaton(rule)
                    state = current_state.copy()
                    while len(state) > 1:
                        state = automaton.apply_rule(state)
                    new_state.append(state[0] if state else 0)
                current_state = new_state
            actual_output = current_state

            if self.visualization_window is None or not self.visualization_window.winfo_exists():
                self.visualization_window = tk.Toplevel(self.master)
                from visualization.network_visualizer import NetworkVisualizer
                self.visualization_canvas = NetworkVisualizer(
                    self.visualization_window,
                    input_size,
                    processing_layers,
                    initial_state,
                    actual_output,
                    expected_output
                )
            else:
                if self.visualization_canvas is not None:
                    self.visualization_canvas.update_network(
                        processing_layers,
                        initial_state,
                        actual_output,
                        expected_output
                    )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to visualize network: {str(e)}")

    def on_new_test_set_created(self):
        if self.visualization_window and self.visualization_window.winfo_exists():
            self.visualization_window.destroy()
        self.visualize_current_network()

    def generate_test_set(self, test_set_number):
        with open(f"test_set_{test_set_number}.txt", "w") as file:
            file.write("Test set data")
        print(f"Generated test_set_{test_set_number}.txt")
        self.on_new_test_set_created()

    def toggle_crossover_entry(self):
        if self.crossover_check_var.get():
            self.crossover_percent_entry.config(state='normal')
        else:
            self.crossover_percent_entry.config(state='disabled')
            self.crossover_percent_var.set("20")

    def toggle_dynamic_adjustment(self):
        if self.dynamic_adjust_var.get():
            self.show_dynamic_adjustment_window()
        else:
            if hasattr(self, 'dynamic_adjustment_confirmed') and not self.dynamic_adjustment_confirmed:
                self.min_percentage_threshold = 0.00
            self.hide_dynamic_adjustment_window()

    def show_dynamic_adjustment_window(self):
        if self.dynamic_adjustment_window is None or not self.dynamic_adjustment_window.winfo_exists():
            self.dynamic_adjustment_window = tk.Toplevel(self.master)
            self.dynamic_adjustment_window.title("Dynamic Adjustment Settings")
            self.dynamic_adjustment_window.protocol("WM_DELETE_WINDOW", self.on_dynamic_window_close)
            self.dynamic_adjustment_confirmed = False

            self.dynamic_adjustment_label = tk.Label(self.dynamic_adjustment_window, text="Minimum Percentage Threshold (%):")
            self.dynamic_adjustment_label.pack(padx=10, pady=5)
            self.dynamic_adjustment_entry = tk.Entry(self.dynamic_adjustment_window, validate="key")
            self.dynamic_adjustment_entry.pack(padx=10, pady=5)
            self.dynamic_adjustment_entry.insert(0, str(int(self.min_percentage_threshold * 100)))
            confirm_btn = tk.Button(self.dynamic_adjustment_window, text="Confirm", command=self.save_dynamic_adjustment)
            confirm_btn.pack(pady=5)
            self.dynamic_adjustment_entry['validatecommand'] = (self.dynamic_adjustment_entry.register(self.validate_percentage), '%P')

    def save_dynamic_adjustment(self):
        try:
            value = int(self.dynamic_adjustment_entry.get())
            if 0 <= value <= 100:
                self.min_percentage_threshold = value / 100.0
                self.dynamic_adjustment_confirmed = True
                self.hide_dynamic_adjustment_window()
            else:
                messagebox.showerror("Error", "The value must be between 0 and 100")
        except ValueError:
            messagebox.showerror("Error", "Invalid numeric value")

    def on_dynamic_window_close(self):
        if not self.dynamic_adjustment_confirmed:
            self.min_percentage_threshold = 0.00
        self.hide_dynamic_adjustment_window()

    def hide_dynamic_adjustment_window(self):
        if self.dynamic_adjustment_window and self.dynamic_adjustment_window.winfo_exists():
            self.dynamic_adjustment_window.withdraw()

    def validate_numeric(self, value):
        return value.isdigit() and 1 <= int(value) <= 20 or value == ""

    def validate_percentage(self, value):
        if value == "": return True
        if not value.isdigit(): return False
        return 0 <= int(value) <= 100

    def validate_crossover_percent(self, value):
        if value == "": return True
        if not value.isdigit(): return False
        return 0 <= int(value) <= 100

    def validate_random_num(self, value):
        if value == "": return True
        if not value.isdigit(): return False
        num = int(value)
        return 1 <= num <= 20

    def validate_layers(self, value):
        return value.isdigit() and 1 <= int(value) <= 10 or value == ""

    def validate_size(self, value):
        return value.isdigit() or value == ""

    def validate_expected_result(self, value):
        return all(c in "01" for c in value) or value == ""

    def validate_initial_state(self, value):
        if value == "": return True
        return all(c in "01," for c in value)

    def validate_initial_states(self, initial_states):
        expected_length = self.size_entry.get()
        if expected_length:
            expected_length = int(expected_length)
            for state in initial_states:
                if len(state) != expected_length:
                    messagebox.showerror("Error",f"Each initial state must have length {expected_length}. Invalid state: {state}")
                    return False
        else:
            lengths = [len(state) for state in initial_states]
            if len(set(lengths)) != 1:
                messagebox.showerror("Error", "All initial states must have the same length.")
                return False
        return True

    def validate_rules(self, value):
        try:
            rules = [int(rule.strip()) for rule in value.split(",") if rule.strip().isdigit()]
            return all(0 <= rule <= 255 for rule in rules)
        except ValueError:
            return False

    def update_initial_state_input(self):
        if self.initial_state_method.get() == "manual":
            self.random_num_entry.grid_remove()
            self.random_num_entry_label.grid_remove()
            self.random_num_entry.delete(0, tk.END)
            self.random_num_entry.config(state="disabled")
            self.initial_state_entry.delete(0, tk.END)
        elif self.initial_state_method.get() == "random":
            self.random_num_entry.grid()
            self.random_num_entry_label.grid()
            self.random_num_entry.config(state="normal")
            self.initial_state_entry.delete(0, tk.END)
        else:
            self.random_num_entry.grid_remove()
            self.random_num_entry_label.grid_remove()
            self.random_num_entry.delete(0, tk.END)
            self.random_num_entry.config(state="disabled")
            self.initial_state_entry.delete(0, tk.END)

    def update_rules_input(self):
        self.hide_dynamic_adjustment_window()
        for widget in self.dynamic_input_frame.winfo_children():
            widget.destroy()

        input_method = self.rules_input_method.get()
        if input_method == "manual":
            hidden_layers = int(self.hidden_layers_entry.get()) if self.hidden_layers_entry.get().isdigit() else 1
            self.simulation.rules_layers = []
            for i in range(hidden_layers):
                layer_label = tk.Label(self.dynamic_input_frame, text=f"Layer {i + 1} Rules:")
                layer_label.grid(row=i, column=0, padx=5, pady=2, sticky="w")

                rules_entry = tk.Entry(self.dynamic_input_frame, width=30, validate="key")
                rules_entry.grid(row=i, column=1, padx=5, pady=2, sticky="w")
                rules_entry['validatecommand'] = (rules_entry.register(self.validate_rules), '%P')

                self.rules_layers.append(rules_entry)
        elif input_method == "file":
            self.file_button = tk.Button(self.dynamic_input_frame, text="Select File", command=self.load_rules_from_file)
            self.file_button.grid(row=0, column=0, sticky="w")
        elif input_method == "random":
            self.random_button = tk.Button(self.dynamic_input_frame, text="Generate Random Rules", command=self.generate_random_rules)
            self.random_button.grid(row=0, column=0, sticky="w")

    def setup_training_tab(self):
        self.load_btn = tk.Button(self.training_tab, text="Load Training File", command=self.load_training_file)
        self.process_btn = tk.Button(self.training_tab, text="Process Data", command=self.process_training_data, state="disabled")
        self.visualize_btn = tk.Button(self.training_tab, text="Visualize Success", command=self.visualize_success, state="disabled")
        self.export_btn = tk.Button(self.training_tab, text="Export Plot", command=self.export_training_plot, state="disabled")
        self.load_btn.grid(row=0, column=0, padx=5, pady=5)
        self.process_btn.grid(row=0, column=1, padx=5, pady=5)
        self.visualize_btn.grid(row=0, column=2, padx=5, pady=5)
        self.export_btn.grid(row=0, column=3, padx=5, pady=5)

    def load_training_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not file_path:
            return
        try:
            n = int(self.size_entry.get())
            with open(file_path, 'r') as f:
                self.training_pairs = []
                errors = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if ',' not in line:
                        errors.append(f"Line {line_num}: No commas")
                        continue
                    input_str, expected_str = map(str.strip, line.split(',', 1))

                    if len(input_str) != n or len(expected_str) != n:
                        errors.append(f"Line {line_num}: Invalid length. Expected {n} characters")
                        continue
                    if not all(c in '01' for c in input_str) or not all(c in '01' for c in expected_str):
                        errors.append(f"Line {line_num}: Non-binary values")
                        continue
                    self.training_pairs.append((input_str, expected_str))

                if errors:
                    messagebox.showerror("Errors in the file", "\n".join(errors[:3] + ["..."] if len(errors) > 3 else errors))
                    return
                self.process_btn.config(state="normal")
                messagebox.showinfo("Success", f"Loaded {len(self.training_pairs)} valid training pairs")
        except ValueError:
            messagebox.showerror("Error", "Invalid 'Size of Each Sequence' value")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def get_best_test_set_rules(self):
        test_set_files = []
        test_set_number = 1
        while os.path.exists(f"test_set_{test_set_number}.txt"):
            test_set_files.append(f"test_set_{test_set_number}.txt")
            test_set_number += 1

        rule_set_counts = defaultdict(int)
        best_rules = None
        max_count = 0

        for file_path in test_set_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    ref_match = re.search(r'Rules from (test_set_\d+)', content)
                    if ref_match:
                        ref_file = ref_match.group(1) + ".txt"
                        try:
                            with open(ref_file, 'r') as ref_f:
                                ref_content = ref_f.read()
                                rules_section = ref_content.split("Rules:\n")[1].split("\n\nRule Scores:")[0]
                        except FileNotFoundError:
                            continue
                    else:
                        rules_section = content.split("Rules:\n")[1].split("\n\nRule Scores:")[0]

                    layers = []
                    for line in rules_section.split('\n'):
                        line = line.strip()
                        if line.startswith('///////////////'):
                            continue
                        try:
                            layers.append(list(map(int, line.split(','))))
                        except (ValueError, IndexError):
                            continue

                    signature = tuple(tuple(layer) for layer in layers)
                    rule_set_counts[signature] += 1

                    if rule_set_counts[signature] > max_count:
                        max_count = rule_set_counts[signature]
                        best_rules = layers
            except Exception as e:
                continue

        return best_rules

    def process_training_data(self):
        if not hasattr(self, 'training_pairs'):
            messagebox.showerror("Error", "No training data loaded")
            return
        print("\n" + "=" * 50)
        best_rules = self.get_best_test_set_rules()
        if best_rules is None:
            best_rules = self.simulation.rules_layers
            messagebox.showinfo("Info", "No test sets found. Using current rules.")
        else:
            messagebox.showinfo("Info", "Using the best test set rules based on frequency.")

        print("Using rules:")
        for i, layer in enumerate(best_rules):
            print(f"Layer {i + 1}: {layer}")
        print("=" * 50 + "\n")

        try:
            n = int(self.size_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid 'Size of Each Sequence' value")
            return
        results = []
        valid_pairs = 0

        for input_str, expected_str in self.training_pairs:
            if len(input_str) != n or len(expected_str) != n:
                results.append(f"{input_str},{expected_str} - INVALID LENGTH")
                continue
            try:
                input_state = [int(c) for c in input_str]
                expected_output = [int(c) for c in expected_str]

                current_state = input_state.copy()
                for layer in best_rules:
                    new_state = []
                    for rule in layer:
                        automaton = CellularAutomaton(rule)
                        state = current_state.copy()
                        while len(state) > 1:
                            state = automaton.apply_rule(state)
                        new_state.append(state[0] if state else 0)
                    current_state = new_state

                result = "+" if current_state == expected_output else "-"
                results.append(f"{input_str},{expected_str} {result}")
                valid_pairs += 1
            except Exception as e:
                results.append(f"{input_str},{expected_str} - ERROR: {str(e)}")

        with open("training_output.txt", "w") as f:
            f.write("\n".join(results))

        self.visualize_btn.config(state="normal")
        self.export_btn.config(state="disabled")
        messagebox.showinfo("Success", f"Processed {valid_pairs}/{len(self.training_pairs)} valid cases")

    def visualize_success(self):
        try:
            with open("training_output.txt", "r") as f:
                results = [line.strip().endswith('+') for line in f]
        except FileNotFoundError:
            messagebox.showerror("Error", "No results file found")
            return

        success = sum(results)
        total = len(results)
        self.current_training_fig = Figure(figsize=(5, 5))
        ax = self.current_training_fig.add_subplot(111)
        ax.pie([success, total - success],
               labels=['Success', 'Failure'],
               autopct='%1.1f%%',
               startangle=90,
               colors=['#4CAF50', '#F44336'])
        ax.set_title("Success Rate")

        window = tk.Toplevel(self.master)
        canvas = FigureCanvasTkAgg(self.current_training_fig, window)
        canvas.get_tk_widget().pack()
        toolbar = NavigationToolbar2Tk(canvas, window)
        toolbar.update()

        self.export_btn.config(state="normal")

    def export_training_plot(self):
        if not hasattr(self, 'current_training_fig') or self.current_training_fig is None:
            messagebox.showwarning("Warning", "No plot to export. Please generate a plot first.")
            return

        file_types = [
            ('PDF', '*.pdf'),
            ('PNG', '*.png'),
            ('SVG', '*.svg')
        ]
        path = filedialog.asksaveasfilename(
            filetypes=file_types,
            defaultextension=".pdf",
            title="Save Plot As"
        )

        if not path:
            return

        try:
            self.current_training_fig.savefig(path, bbox_inches='tight', dpi=300)
            messagebox.showinfo("Success", f"Plot exported to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")

    def load_initial_state_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not file_path:
            return
        try:
            with open(file_path, 'r') as file:
                initial_states = [line.strip() for line in file if line.strip()]
                initial_states_str = ','.join(initial_states)
                self.initial_state_entry.delete(0, tk.END)
                self.initial_state_entry.insert(0, initial_states_str)
                self.validate_initial_states(initial_states)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load initial state from file: {str(e)}")

    def load_rules_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if not file_path:
            return
        try:
            with open(file_path, "r") as file:
                self.rules_from_file = []
                for line in file:
                    line = line.strip()
                    if line:
                        rules = [int(rule) for rule in line.split(",") if rule.isdigit()]
                        if rules:
                            self.rules_from_file.append(rules)
                if self.rules_from_file:
                    self.simulation.rules_layers = self.rules_from_file
                    self.hidden_layers_entry.delete(0, tk.END)
                    self.hidden_layers_entry.insert(0, len(self.rules_from_file))
                    messagebox.showinfo("Success", f"Loaded {len(self.rules_from_file)} layers from file.")
                else:
                    raise ValueError("File does not contain valid rules.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load rules from file: {e}")

    def generate_random_bit_sequences(self, num_sequences, sequence_size):
        sequences = ','.join(''.join(random.choice('01') for _ in range(sequence_size)) for _ in range(num_sequences))
        self.temp_initial_state = sequences
        return sequences

    def generate_random_rules(self):
        try:
            size = int(self.size_entry.get())
            hidden_layers = int(self.hidden_layers_entry.get())
            self.rules_from_random = [[random.randint(0, 255) for _ in range(size)] for _ in range(hidden_layers)]
            self.simulation.rules_layers = self.rules_from_random
            messagebox.showinfo("Success", f"Generated random rules for {hidden_layers} layers.")
        except ValueError:
            messagebox.showerror("Error","Invalid input. Please provide valid numbers for 'Size of each sequence' and 'Hidden Layers'.")

    def handle_empty_initial_state(self):
        self.initial_state_entry.config(state='disabled')
        self.master.after(1, self.restore_initial_state)

    def restore_initial_state(self):
        self.initial_state_entry.config(state='normal')
        self.initial_state_entry.delete(0, tk.END)
        self.initial_state_entry.insert(0, self.temp_initial_state)
        del self.temp_initial_state

    def train_until_success(self):
        if self.dynamic_adjust_var.get() and self.dynamic_iteration_count < 1:
            self.dynamic_iteration_count = 0
        self.is_step_by_step = False
        self.temp_initial_state = self.initial_state_entry.get()
        initial_state = self.initial_state_entry.get().strip()
        if not self.validate_initial_state(initial_state):
            messagebox.showwarning("Warning", "Initial state must contain only 0, 1, or commas. Training will be aborted.")
            return

        input_method = self.rules_input_method.get()
        if input_method == "manual":
            try:
                n = int(self.size_entry.get())
                self.simulation.rules_layers = []
                for entry in self.rules_layers:
                    rules_input = entry.get().strip()
                    if rules_input:
                        rules = [int(rule.strip()) for rule in rules_input.split(",") if rule.strip().isdigit()]
                        if len(rules) != n:
                            raise ValueError(f"Number of rules in layer must be equal {n}.")
                        self.simulation.rules_layers.append(rules)
            except ValueError as e:
                messagebox.showerror("Input Error", str(e))
                return
        elif input_method == "random" and not self.simulation.rules_layers:
            self.generate_random_rules()
        if not self.simulation.rules_layers or any(not layer for layer in self.simulation.rules_layers):
            messagebox.showerror("Input Error", "No rules available. Please provide rules.")
            return

        self.visualize_check_state_before_training = self.visualize_network_var.get()
        self.visualize_network_var.set(False)
        self.visualize_check.config(state="disabled")
        self.step_button.config(state="disabled")
        self.run_training_loop(self.simulation.rules_layers)

    def run_training_loop(self, rules_layers):
        iteration = 0
        max_iterations = 100000

        def training_step():
            nonlocal iteration
            self.run_simulation()
            self.update_missing_test_set()

            if self.check_test_set_exists(self.missing_test_set):
                print(f"Success: Found all necessary test sets up to test_set_{self.missing_test_set - 1}.")
                self.step_button.config(state="normal")
                self.visualize_check.config(state="normal")
                if self.visualize_check_state_before_training:
                    self.visualize_check.select()
                return

            if not self.initial_state_entry.get().strip():
                print("All test sets generated, stopping training.")
                self.step_button.config(state="normal")
                self.visualize_check.config(state="normal")
                if self.visualize_check_state_before_training:
                    self.visualize_check.select()
                return

            self.train_network()
            iteration += 1
            if iteration < max_iterations:
                self.master.after(0, training_step)
            else:
                messagebox.showinfo("Info", "Training did not converge within max iterations.")
                self.step_button.config(state="normal")
                self.visualize_check.config(state="normal")
                if self.visualize_check_state_before_training:
                    self.visualize_check.select()
        training_step()

    def on_train_and_replace_click(self):
        self.generate_test_set(self.missing_test_set)
        self.update_missing_test_set()

    def update_missing_test_set(self):
        test_set_number = 1
        while os.path.exists(f"test_set_{test_set_number}.txt"):
            test_set_number += 1
        self.missing_test_set = test_set_number

    def check_test_set_exists(self, test_set_number):
        return os.path.exists(f"test_set_{test_set_number}.txt")

    def train_network(self):
        try:
            rule_scores = self.simulation.get_rule_scores()
            new_rules_layers = self.replace_negative_rules(rule_scores, self.simulation.rules_layers)
            if self.rules_input_method.get() == "manual":
                for i, entry in enumerate(self.rules_layers):
                    entry.delete(0, tk.END)
                    entry.insert(0, ",".join(map(str, new_rules_layers[i])))
            self.simulation.rules_layers = new_rules_layers
            if self.visualize_network_var.get():
                self.visualize_current_network()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_rule_scores(self, rule_scores):
        if not self.is_step_by_step:
            return
        for rule in range(256):
            if rule not in rule_scores:
                rule_scores[rule] = 0
        sorted_rules = sorted(rule_scores.items(), key=lambda x: x[1], reverse=True)
        print("Rules sorted by score:")
        for rule, score in sorted_rules:
            print(f"Rule: {rule}, Score: {score}")
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "Rules sorted by score:\n")
        for rule, score in sorted_rules:
            self.results_text.insert(tk.END, f"Rule: {rule}, Score: {score}\n")
        self.results_text.config(state="disabled")

    def replace_negative_rules(self, rule_scores, current_rules_layers):
        if self.simulation.successful_run:
            self.simulation.successful_run = False
            return current_rules_layers
        new_rules_layers = []
        if self.crossover_check_var.get():
            try:
                base_crossover = int(self.crossover_percent_entry.get()) / 100.0
            except ValueError:
                base_crossover = 0.2
        else:
            base_crossover = 0.2

        if self.dynamic_adjust_var.get():
            adjustment = self.dynamic_iteration_count * 0.001
            crossover_probability = base_crossover - adjustment
            crossover_probability = max(crossover_probability, self.min_percentage_threshold)
            try:
                min_percentage_threshold = int(self.dynamic_adjustment_entry.get()) / 100.0
            except ValueError:
                min_percentage_threshold = 0.00

            crossover_probability = max(crossover_probability, min_percentage_threshold)
            if self.is_step_by_step:
                print("\n--- Dynamic Adjustment Debug Info ---")
                print(f"Base Crossover: {base_crossover}")
                print(f"Iteration Count: {self.dynamic_iteration_count}")
                print(f"Adjustment: {adjustment}")
                print(f"Final Crossover Probability: {crossover_probability}")
            self.dynamic_iteration_count += 1
        else:
            crossover_probability = base_crossover
            if self.is_step_by_step:
                print("\n--- Crossover Debug Info ---")
                print(f"Static Crossover Probability: {crossover_probability}")
        sorted_rules = sorted(rule_scores.items(), key=lambda x: x[1], reverse=True)
        top_rules = [r for r, s in sorted_rules[:25]]

        for layer in current_rules_layers:
            new_layer_rules = []
            for rule in layer:
                if rule_scores.get(rule, 0) < 0 or random.random() < crossover_probability:
                    if top_rules and random.random() < 0.7:
                        new_rule = random.choice(top_rules)
                    else:
                        mutation_range = max(1, int(255 * (1 - rule_scores.get(rule, 0) / 100)))
                        new_rule = (rule + random.randint(-mutation_range, mutation_range)) % 256
                        if new_rule in [0, 255] and rule_scores.get(new_rule, 0) < 50:
                            new_rule = (new_rule + random.randint(1, 254)) % 256
                else:
                    improvement = random.gauss(0, 10)
                    new_rule = min(255, max(0, int(rule + improvement)))
                    if new_rule in [0, 255] and rule_scores.get(new_rule, 0) < 50:
                        new_rule = (new_rule + random.randint(1, 254)) % 256
                new_layer_rules.append(new_rule)
            new_rules_layers.append(new_layer_rules)
        return new_rules_layers

    def load_rule_scores(self, file_path):
        rules_scores = defaultdict(int)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith("##### Rule"):
                    rule = int(line.split("##### Rule")[1].strip())
                    if i + 2 < len(lines):
                        result_line = lines[i + 2].strip()
                        if result_line.startswith("Result:"):
                            result = int(result_line.split(":")[1].strip())
                            rules_scores[rule] += result
        return rules_scores

    def run_simulation(self):
        try:
            hidden_layers = self.hidden_layers_entry.get()
            if not hidden_layers.isdigit():
                raise ValueError("Hidden Layers must be a number.")
            hidden_layers = int(hidden_layers)
            n = self.size_entry.get()
            if not n.isdigit() or int(n) < 1:
                raise ValueError("Size of each sequence must be a positive number.")
            n = int(n)
            if n % 2 == 0 or n < 3 or n > 101:
                raise ValueError("Size of each sequence must be an odd number between 3 and 101.")
            if self.rules_input_method.get() == "manual":
                self.simulation.rules_layers = []
                for entry in self.rules_layers:
                    rules_input = entry.get().strip()
                    if rules_input:
                        rules = [int(rule.strip()) for rule in rules_input.split(",") if rule.strip().isdigit()]
                        if len(rules) != n:
                            raise ValueError(f"Number of rules in layer must be equal {n}.")
                        self.simulation.rules_layers.append(rules)

            if self.initial_state_method.get() == "manual" and self.initial_state_entry.get().strip():
                if self.visualize_network_var.get():
                    self.visualize_current_network()
                initial_state = self.initial_state_entry.get()
                initial_states = [state.strip() for state in initial_state.split(",")]
                if not initial_states or len(initial_states) == 0:
                    raise ValueError("No valid initial state provided.")
                for state in initial_states:
                    if len(state) != n:
                        raise ValueError(f"Initial State must be exactly {n} binary digits (0 or 1).")
                initial_state = [int(bit) for bit in initial_states[0]]
            else:
                if self.initial_state_method.get() in ["manual", "file"] and self.initial_state_entry.get().strip():
                    if self.visualize_network_var.get():
                        self.visualize_current_network()
                    initial_state = self.initial_state_entry.get()
                    initial_states = [state.strip() for state in initial_state.split(",")]
                    if not initial_states or len(initial_states) == 0:
                        raise ValueError("No valid initial state provided.")
                    for state in initial_states:
                        if len(state) != n:
                            raise ValueError(f"Initial State must be exactly {n} binary digits (0 or 1).")
                    initial_state = [int(bit) for bit in initial_states[0]]
                elif self.initial_state_method.get() == "random":
                    current_entry = self.initial_state_entry.get().strip()
                    if current_entry:
                        initial_states = current_entry.split(',')
                        initial_state = [int(bit) for bit in initial_states[0].strip()]
                    else:
                        random_num = int(self.random_num_entry.get())
                        sequence_size = int(self.size_entry.get())
                        random_initial_state = self.generate_random_bit_sequences(random_num, sequence_size)
                        self.initial_state_entry.delete(0, tk.END)
                        self.initial_state_entry.insert(0, random_initial_state)
                        initial_states = random_initial_state.split(',')
                        initial_state = [int(bit) for bit in initial_states[0].strip()]
                    if self.visualize_network_var.get():
                        self.visualize_current_network()
                else:
                    if self.initial_state is None:
                        self.initial_state = [random.choice([0, 1]) for _ in range(n)]
                        self.initial_state_entry.config(state="normal")
                        self.initial_state_entry.delete(0, tk.END)
                        self.initial_state_entry.insert(0, "".join(map(str, self.initial_state)))
                    initial_state = self.initial_state

            expected_result = self.expected_result_entry.get().strip()
            if not expected_result:
                raise ValueError("Expected result cannot be empty.")
            if not all(c in "01" for c in expected_result):
                raise ValueError("Expected result must only contain 0 and 1.")
            expected_result = [int(bit) for bit in expected_result]

            rules_layers = self.simulation.rules_layers
            if not rules_layers:
                raise ValueError("No rules available. Please train or provide rules.")

            self.simulation.n = n
            self.simulation.hidden_layers = hidden_layers

            if n <= 13:
                final_state, hamming_distance = self.simulation.run(
                    rules_layers=rules_layers,
                    initial_state=initial_state,
                    expected_result=expected_result
                )
            else:
                final_state, hamming_distance = self.simulation.run_in_memory(
                    rules_layers=rules_layers,
                    initial_state=initial_state,
                    expected_result=expected_result
                )
            if hamming_distance == 0 and self.visualize_network_var.get():
                self.visualize_current_network()
            if self.is_step_by_step:
                history_content = "No history available."
                results_content = "No results available."
                try:
                    with open("history.txt", "r") as history_file:
                        history_content = history_file.read()
                    with open("results.txt", "r") as results_file:
                        results_content = results_file.read()
                except FileNotFoundError:
                    pass

                self.history_text.config(state="normal")
                self.history_text.delete("1.0", tk.END)
                self.history_text.insert(tk.END, f"Input:\n{history_content}")
                self.history_text.config(state="disabled")

                self.results_text.config(state="normal")
                self.results_text.delete("1.0", tk.END)
                self.results_text.insert(tk.END, f"Output:\n{results_content.strip()}")
                self.results_text.config(state="disabled")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

    def append_no_training_message(self, file_path):
        with open(file_path, 'a') as f:
            f.write("\nDID NOT REQUIRE TRAINING")

    def run_test_set_files(self):
        self.is_running_test_set = True
        try:
            iterations_str = self.iterations_label_entry.get().strip()
            if not iterations_str or not iterations_str.isdigit():
                messagebox.showerror("Error", "Invalid number of iterations. Please enter a number between 1 and 20.")
                return
            iterations = int(iterations_str)

            initial_state_str = self.initial_state_entry.get().strip()
            if not initial_state_str:
                messagebox.showerror("Error", "Initial state is empty.")
                return
            initial_states = [state.strip() for state in initial_state_str.split(",") if state.strip()]

            expected_result_str = self.expected_result_entry.get().strip()
            if not expected_result_str:
                messagebox.showerror("Error", "Expected result is empty.")
                return
            expected_result = [int(c) for c in expected_result_str]
            current_states = initial_states.copy()

            for iteration in range(iterations):
                new_states = []
                for idx, initial_state_str in enumerate(current_states):
                    if not all(c in '01' for c in initial_state_str):
                        messagebox.showerror("Error", f"Invalid initial state: {initial_state_str}")
                        continue
                    initial_state = [int(c) for c in initial_state_str]
                    test_set_number = 1
                    while os.path.exists(f"test_set_{test_set_number}.txt"):
                        test_set_number += 1
                    file_path = f"test_set_{test_set_number}.txt"

                    prev_test_set_number = test_set_number - 1
                    prev_rules = []
                    while prev_test_set_number >= 1:
                        prev_file = f"test_set_{prev_test_set_number}.txt"
                        if os.path.exists(prev_file):
                            prev_rules = self.get_latest_rules_from_file(prev_file)
                            if prev_rules:
                                break
                        prev_test_set_number -= 1
                    if not prev_rules:
                        prev_rules = self.simulation.rules_layers
                    try:
                        original_rules = self.simulation.rules_layers
                        self.simulation.rules_layers = prev_rules
                        final_state, hamming = self.simulation.run(
                            rules_layers=prev_rules,
                            initial_state=initial_state,
                            expected_result=expected_result
                        )
                    except Exception as e:
                        messagebox.showerror("Error", f"Initial check failed for {file_path}: {e}")
                        self.simulation.rules_layers = original_rules
                        continue
                    finally:
                        self.simulation.rules_layers = original_rules

                    if hamming == 0:
                        with open(file_path, 'w') as f:
                            f.write(f"Input: {initial_state_str}\n")
                            f.write(f"Expected output: {expected_result_str}\n")
                            f.write("Rules:\n")
                            for layer in prev_rules:
                                f.write(",".join(map(str, layer)) + "\n")
                            if prev_test_set_number >= 1:
                                f.write(
                                    f"\n/////////////// Rules from test_set_{prev_test_set_number} solve expected output. ///////////////\n")
                            else:
                                f.write("\nDID NOT REQUIRE TRAINING\n")
                        new_states.append(initial_state_str)
                        continue
                    current_rules = prev_rules
                    max_iterations = 100000
                    trained = False

                    for _ in range(max_iterations):
                        try:
                            self.simulation.rules_layers = current_rules
                            final_state, hamming = self.simulation.run(
                                rules_layers=current_rules,
                                initial_state=initial_state,
                                expected_result=expected_result
                            )
                        except Exception as e:
                            messagebox.showerror("Error", f"Training failed for {file_path}: {e}")
                            break

                        if hamming == 0:
                            self.write_test_set_file(file_path, initial_state, expected_result, current_rules, self.simulation.get_rule_scores())
                            trained = True
                            break

                        rule_scores = self.simulation.get_rule_scores()
                        new_rules_layers = self.replace_negative_rules(rule_scores, current_rules)
                        if new_rules_layers == current_rules and _ > 50:
                            break
                        current_rules = new_rules_layers
                    if trained:
                        self.write_test_set_file(file_path, initial_state, expected_result, current_rules, self.simulation.get_rule_scores())
                    else:
                        self.write_test_set_file(file_path, initial_state, expected_result, current_rules, self.simulation.get_rule_scores())
                        messagebox.showwarning("Warning", f"Training did not fully converge for {file_path}.")
                    new_states.append(initial_state_str)
                current_states = new_states.copy()
        finally:
            self.is_running_test_set = False
        messagebox.showinfo("Success", f"Completed {iterations} iterations!")

    def write_test_set_file(self, file_path, initial_state, expected_result, rules_layers, rule_scores):
        with open(file_path, 'w') as f:
            f.write(f"Input: {''.join(map(str, initial_state))}\n")
            f.write(f"Expected output: {''.join(map(str, expected_result))}\n")
            f.write("Rules:\n")
            for layer in rules_layers:
                f.write(",".join(map(str, layer)) + "\n")
            f.write("\nRule Scores:\n")
            sorted_rules = sorted(rule_scores.items(), key=lambda x: x[1], reverse=True)
            for rule, score in sorted_rules:
                f.write(f"Rule: {rule}, Score: {score}\n")

    def parse_test_set_file(self, file_path):
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return None, None
        initial_state = None
        expected_result = None

        for line in lines:
            line = line.strip()
            if line.startswith("Input:"):
                input_str = line.split("Input: ")[1].strip()
                initial_state = [int(c) for c in input_str]
            elif line.startswith("Expected output:"):
                expected_str = line.split("Expected output: ")[1].strip()
                expected_result = [int(c) for c in expected_str]
            if initial_state and expected_result:
                break
        return initial_state, expected_result

    def parse_test_set_1(self):
        try:
            with open("test_set_1.txt", "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return None, None
        initial_state = None
        expected_result = None

        for line in lines:
            line = line.strip()
            if line.startswith("Input:"):
                input_str = line.split("Input: ")[1].strip()
                initial_state = [int(c) for c in input_str]
            elif line.startswith("Expected output:"):
                expected_str = line.split("Expected output: ")[1].strip()
                expected_result = [int(c) for c in expected_str]
            if initial_state and expected_result:
                break
        return initial_state, expected_result

    def get_latest_test_set_number(self):
        test_set_number = 1
        while os.path.exists(f"test_set_{test_set_number}.txt"):
            test_set_number += 1
        latest = test_set_number - 1
        return latest if latest >= 1 else None

    def get_latest_rules_from_file(self, file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        rules_section = content.split('///////////////')[0].strip()
        rules = []
        in_rules = False
        for line in rules_section.split('\n'):
            line = line.strip()
            if line.startswith('Rules:'):
                in_rules = True
                continue
            if in_rules:
                if line.startswith('Rule Scores:') or line.startswith('DID NOT REQUIRE TRAINING'):
                    break
                if line and ',' in line:
                    rules.append([int(num) for num in line.split(',')])
        return rules if rules else None

    def append_to_test_set(self, file_path, initial_state, expected_result, rules_layers, rule_scores, hamming):
        previous_rules = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                sections = content.split('// // // // // // // // // // // // // // // // // // // // // // // //')
                if sections:
                    last_section = sections[-1].strip()
                    if last_section:
                        rules_start = last_section.find("Rules:\n")
                        if rules_start != -1:
                            rules_part = last_section[rules_start + len("Rules:\n"):]
                            rules_lines = [line.strip() for line in rules_part.split('\n') if line.strip()]
                            previous_rules = []
                            for line in rules_lines:
                                if line.startswith("Rule Scores:") or line.startswith("DID NOT REQUIRE TRAINING"):
                                    break
                                if ',' in line:
                                    previous_rules.append([int(num) for num in line.split(',')])
        rules_identical = previous_rules and (rules_layers == previous_rules)

        if rules_identical:
            with open(file_path, 'a') as f:
                f.write("\n\nDID NOT REQUIRE TRAINING")
        else:
            with open(file_path, 'a') as test_file:
                test_file.write("\n// // // // // // // // // // // // // // // // // // // // // // // //\n")
                test_file.write(f"Input: {''.join(map(str, initial_state))}\n")
                test_file.write(f"Expected output: {''.join(map(str, expected_result))}\n")
                test_file.write("Rules:\n")
                for layer in rules_layers:
                    test_file.write(",".join(map(str, layer)) + "\n")
                test_file.write("\nRule Scores:\n")
                sorted_rules = sorted(rule_scores.items(), key=lambda item: item[1], reverse=True)
                for rule, score in sorted_rules:
                    test_file.write(f"Rule: {rule}, Score: {score}\n")
                if hamming == 0:
                    test_file.write("Correct!\n")

    def append_note_to_test_set(self, file_path, new_test_set_number, source_test_set_file):
        note = f"\n\n/////////////// Rules from test_set_{new_test_set_number} solve expected output with initial state from {source_test_set_file}. ///////////////"
        with open(file_path, 'a') as f:
            f.write(note)