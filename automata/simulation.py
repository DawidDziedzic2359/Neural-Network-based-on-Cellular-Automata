import os
import random
from collections import defaultdict
from automata.cellular_automaton import CellularAutomaton
import tkinter as tk
from tkinter import messagebox

class Simulation:
    def __init__(self, gui=None, n=10, hidden_layers=0):
        self.gui = gui
        self.n = n
        self.hidden_layers = hidden_layers
        self.rules_layers = []
        self.all_states = []
        self.rule_scores = defaultdict(int)
        self.successful_run = False
        self.log_buffer = []
        self.iteration_counter = 0

    def load_rule_scores_from_file(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith("Rule: "):
                    parts = line.split(", Score: ")
                    if len(parts) == 2:
                        rule = int(parts[0].replace("Rule: ", "").strip())
                        score = int(parts[1].strip())
                        self.rule_scores[rule] = score

    def generate_initial_state(self):
        return [random.choice([0, 1]) for _ in range(self.n)]

    def hamming_distance(self, state1, state2):
        return sum(1 for a, b in zip(state1, state2) if a != b)

    def save_test_results(self, initial_state, expected_result, rules_layers):
        test_set_number = 1
        while os.path.exists(f"test_set_{test_set_number}.txt"):
            test_set_number += 1
        with open(f"test_set_{test_set_number}.txt", "w") as test_file:
            test_file.write(f"Input: {''.join(map(str, initial_state))}\n")
            test_file.write(f"Expected output: {''.join(map(str, expected_result))}\n")
            test_file.write("Rules:\n")
            for layer_index, layer_rules in enumerate(rules_layers):
                test_file.write(",".join(map(str, layer_rules)) + "\n")
            test_file.write("\nRule Scores:\n")
            sorted_rules = sorted(self.rule_scores.items(), key=lambda item: item[1], reverse=True)
            for rule, score in sorted_rules:
                test_file.write(f"Rule: {rule}, Score: {score}\n")
        if not self.gui.is_running_test_set:
            messagebox.showinfo("Info", "Learning completed successfully. Trained dataset saved.")

        initial_state_str = self.gui.initial_state_entry.get()
        if "," in initial_state_str:
            parts = initial_state_str.split(",", 1)
            new_initial_state_str = parts[1] if len(parts) > 1 else ""
        else:
            new_initial_state_str = ""
        self.gui.initial_state_entry.delete(0, tk.END)
        self.gui.initial_state_entry.insert(0, new_initial_state_str)
        new_initial_state_str = self.gui.initial_state_entry.get()
        if not new_initial_state_str.strip() and hasattr(self.gui, 'temp_initial_state'):
            self.gui.master.after(0, self.gui.handle_empty_initial_state)
        new_initial_state_str = self.gui.initial_state_entry.get()
        if not new_initial_state_str.strip():
            self.gui.visualize_button.config(state="normal")
            self.gui.test_set_files_button.config(state="normal")
            self.gui.iterations_label_entry.config(state="normal")

    def get_rule_scores(self):
        return dict(self.rule_scores)

    def flush_logs(self):
        with open("history.txt", "a") as history_file, \
                open("results.txt", "a") as results_file, \
                open("rules_log.txt", "a") as rules_file:
            for entry in self.log_buffer:
                history_file.write(entry.get("history", ""))
                results_file.write(entry.get("results", ""))
                rules_file.write(entry.get("rules", ""))
        self.log_buffer = []

    def run(self, rules_layers, initial_state=None, expected_result=None):
        if initial_state is None:
            initial_state = self.generate_initial_state()
        if expected_result is None:
            raise ValueError("Expected result must be provided.")

        current_state = initial_state
        log_entry = {
            "history": "",
            "results": "",
            "rules": ""
        }

        log_entry["history"] += f"Initial State:\t{' '.join(map(str, initial_state))}\n"
        log_entry["history"] += f"Expected Result:\t{' '.join(map(str, expected_result))}\n"
        log_entry["rules"] += "### Rules and Results Log ###\n"
        log_entry["rules"] += f"Initial State:\t{' '.join(map(str, initial_state))}\n"
        log_entry["rules"] += f"Expected Result:\t{' '.join(map(str, expected_result))}\n"

        for layer_index, layer_rules in enumerate(rules_layers):
            log_entry["rules"] += f"\n####### Layer {layer_index + 1} #######\n"
            layer_results = []
            for i, rule in enumerate(layer_rules):
                log_entry["rules"] += f"\n##### Rule {rule}\n"
                automaton = CellularAutomaton(rule)
                state = current_state[:]
                transformation = [state]
                results_block = []

                while len(state) > 1:
                    state = automaton.apply_rule(state)
                    transformation.append(state)
                    results_block.append(' '.join(map(str, state)))
                transformation_result = state[0] if state else 0
                layer_results.append(transformation_result)

                results_block.insert(0, ' '.join(map(str, transformation[0])))
                log_entry["results"] += f"\n##### Rule {rule}\n"
                for index, result in enumerate(results_block):
                    log_entry["results"] += f"{' ' * (index * 2)}{result}\n"

                expected_bit = expected_result[i] if i < len(expected_result) else 0
                result = 1 if transformation_result == expected_bit else -1
                self.rule_scores[rule] += result
                log_entry["rules"] += f"Transformation Result:\t{transformation_result}\n"
                log_entry["rules"] += f"Result:\t{result}\n"

            current_state = layer_results
            log_entry["history"] += f"State after Layer {layer_index + 1}:\t{' '.join(map(str, current_state))}\n"
        final_output = " ".join(map(str, current_state))
        log_entry["rules"] += f"Final Output:\t{final_output}\n"
        log_entry["history"] += f"Final Output:\t{final_output}\n"
        log_entry["history"] += "////////////////////////////////////////////////\n"
        log_entry["results"] += "////////////////////////////////////////////////\n"

        self.log_buffer.append(log_entry)
        self.iteration_counter += 1
        hamming_output_to_expected = self.hamming_distance(current_state, expected_result)
        if self.iteration_counter % 1000 == 0 or hamming_output_to_expected == 0:
            self.flush_logs()
        if hamming_output_to_expected == 0:
            print("Optimal rule set achieved for this test set.")
            for layer in rules_layers:
                for rule in layer:
                    self.rule_scores[rule] += 1
            self.save_test_results(initial_state, expected_result, rules_layers)
            self.successful_run = True
            self.flush_logs()
        else:
            self.successful_run = False
        return current_state, hamming_output_to_expected

    def run_in_memory(self, rules_layers, initial_state=None, expected_result=None):
        if initial_state is None:
            initial_state = self.generate_initial_state()
        if expected_result is None:
            raise ValueError("Expected result must be provided.")

        current_state = initial_state
        for layer_index, layer_rules in enumerate(rules_layers):
            layer_results = []
            for i, rule in enumerate(layer_rules):
                automaton = CellularAutomaton(rule)
                state = current_state[:]
                while len(state) > 1:
                    state = automaton.apply_rule(state)
                transformation_result = state[0] if state else 0
                layer_results.append(transformation_result)
                expected_bit = expected_result[i] if i < len(expected_result) else 0
                result = 1 if transformation_result == expected_bit else -1
                self.rule_scores[rule] += result
            current_state = layer_results

        hamming_output_to_expected = self.hamming_distance(current_state, expected_result)
        if hamming_output_to_expected == 0:
            self.save_test_results(initial_state, expected_result, rules_layers)
            self.successful_run = True
        else:
            self.successful_run = False
        return current_state, hamming_output_to_expected