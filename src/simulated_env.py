# src/simulated_env.py

import json # Import json to load data from tasks.json
import random
import os # Import os for path joining

class SimulatedAndroidWorldEnv:
    def __init__(self, task_data_path):
        self.current_state = {} 
        self.goal = ""
        self.task_data_path = task_data_path
        self.task_data = self._load_task_data()
        self.current_task_info = None

    def _load_task_data(self):
        # Load tasks from the specified JSON file
        try:
            with open(self.task_data_path, "r") as f: 
                data = json.load(f) 
            # Assuming the JSON contains a "tasks" key with a list of task objects
            # Convert list to a dictionary keyed by a unique task identifier (e.g., 'id')
            tasks_dict = {task["id"]: task for task in data.get("tasks", [])}
            return tasks_dict
        except FileNotFoundError:
            print(f"[SimulatedEnv] Error: Task data file '{self.task_data_path}' not found. Using empty task list.")
            return {}
        except json.JSONDecodeError:
            print(f"[SimulatedEnv] Error: Could not decode JSON from '{self.task_data_path}'. Using empty task list.")
            return {}

    def get_available_tasks(self):
        """Returns the dictionary of loaded tasks."""
        return self.task_data

    def reset(self, task_description):
        self.goal = task_description
        
        # Find the relevant task data by matching goal description or ID if provided
        self.current_task_info = None
        for task_id, info in self.task_data.items():
            if task_id == task_description or info["goal"].lower() == task_description.lower():
                self.current_task_info = info
                break
        
        if not self.current_task_info:
            print(f"[SimulatedEnv] Warning: Task '{task_description}' not found in simulated data. Using generic initial state.")
            self.current_state = {"ui_elements": [{"text": "Home Screen", "resource_id": "home_screen"}]}
            self.current_task_info = {
                "initial_observation": self.current_state,
                "optimal_path": [],
                "final_observation": self.current_state,
                "success_condition": lambda obs: False 
            }
        else:
            self.current_state = self.current_task_info["initial_observation"]

        return self.current_state

    def step(self, action):
        reward = 0
        done = False
        info = {}

        action_type = action.get("action_type")
        element_id = action.get("element_id")
        action_text = action.get("text")

        print(f"[SimulatedEnv] Executing simulated action: {action_type} on {element_id} with text '{action_text}'")

        if self.current_task_info:
            next_state_simulated = False
            # This is a very simplistic way to check if an action is "correct"
            # In a real simulation, you'd need more sophisticated state transition logic
            for i, expected_action in enumerate(self.current_task_info["optimal_path"]):
                if action_type == expected_action.get("action_type") and \
                   element_id == expected_action.get("element_id") and \
                   action_text == expected_action.get("text", action_text): # Check text for input actions
                    print(f"[SimulatedEnv] Action matched step {i+1} in optimal path.")
                    
                    if i + 1 < len(self.current_task_info["optimal_path"]):
                        # Simulate some intermediate UI change
                        self.current_state = {"ui_elements": [{"text": f"Screen {i+2}", "resource_id": f"screen_id_{i+2}"}]}
                    else:
                        self.current_state = self.current_task_info["final_observation"]
                    
                    reward = 1.0 
                    next_state_simulated = True
                    break
            
            if not next_state_simulated:
                print(f"[SimulatedEnv] Action {action_type}({element_id}) did not match expected optimal path or lead to a new state.")
                reward = -0.1 

            # Check if goal is reached based on the success condition (assumed to be a lambda function for now)
            # You might need to make the success_condition more robust to string-based checks
            if callable(self.current_task_info["success_condition"]) and self.current_task_info["success_condition"](self.current_state):
                done = True
                reward = 10.0 
                info["success"] = True
                print("[SimulatedEnv] Goal achieved!")
            elif not callable(self.current_task_info["success_condition"]) and self.current_task_info["success_condition"] == "brightness_slider_visible" and \
                "brightness_slider" in [e["resource_id"] for e in self.current_state["ui_elements"]]:
                done = True
                reward = 10.0 
                info["success"] = True
                print("[SimulatedEnv] Goal achieved! (Brightness slider visible)")
            elif not callable(self.current_task_info["success_condition"]) and self.current_task_info["success_condition"] == "calculator_result_is_60" and \
                any(e.get("text") == "60" for e in self.current_state["ui_elements"] if e.get("resource_id") == "result_display"):
                done = True
                reward = 10.0 
                info["success"] = True
                print("[SimulatedEnv] Goal achieved! (Calculator result is 60)")

            elif False: # This condition was a placeholder, needs to be handled properly if max steps is based on env, not loop
                done = True
                info["success"] = False
                print("[SimulatedEnv] Max steps reached. Task failed.")

        else: 
            if action_type == "click" and element_id == "home_screen_icon":
                self.current_state = {"ui_elements": [{"text": "Home", "resource_id": "home_screen_updated"}]}
            elif action_type == "input":
                self.current_state = {"ui_elements": [{"text": f"Inputted: {action_text}", "resource_id": "input_feedback"}]}
            if random.random() < 0.1: 
                done = True
                reward = -5.0
                info["success"] = False
            else:
                reward = -0.5 

        return self.current_state, reward, done, info
