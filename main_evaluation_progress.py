import os
import json
from datetime import datetime

from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer 

from prompts.prompts import ANDROID_AGENT_COT_PROMPT, NAVIGATION_PROMPT, INFORMATION_EXTRACTION_PROMPT 
from src.simulated_env import SimulatedAndroidWorldEnv 

from tqdm import tqdm 

# Configuration
TASK_FILE = "tasks.json" 
BASE_RESULTS_DIR = "results" 

MODELS_TO_COMPARE = [
    "HuggingFaceH4/zephyr-7b-beta",
    "mistralai/Mistral-7B-v0.1"
    # Smaller models optimized for local/CPU inference (often available in quantized versions)
    # "microsoft/Phi-3-mini-4k-instruct",  # A strong smaller model from Microsoft
    # "google/gemma-2b-it", # Google's efficient Gemma 2B instruction-tuned model
    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # A very small model for fast iteration
    # You might need to add other models based on specific needs or latest releases
    # e.g., "stabilityai/stablelm-2-zephyr-1_6b" or other community fine-tunes
]

# ANSI color codes
COLOR_RESET = "\033[0m"
COLOR_BLUE = "\033[34m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_MAGENTA = "\033[35m"
COLOR_CYAN = "\033[36m"
COLOR_RED = "\033[31m"


class HFLLMAgent:
    def __init__(self, model_name, **kwargs): 
        tqdm.write(f"{COLOR_BLUE}Initializing agent with model: {model_name}{COLOR_RESET}") 

        self.generator = pipeline(
            'text-generation', 
            model=model_name, 
            max_new_tokens=256,  
            device_map="auto", 
            trust_remote_code=True, 
            **kwargs
        )
        if self.generator.tokenizer.pad_token is None:
            self.generator.tokenizer.pad_token = self.generator.tokenizer.eos_token
            tqdm.write(f"{COLOR_YELLOW}Warning: Model {model_name} tokenizer did not have a pad token. Set to EOS token.{COLOR_RESET}")

        self.llm = HuggingFacePipeline(pipeline=self.generator)
        tqdm.write(f"{COLOR_BLUE}Model '{model_name}' loaded and ready for inference.{COLOR_RESET}") 

        self.prompts = { 
            "cot_action": ANDROID_AGENT_COT_PROMPT,
            "navigation": NAVIGATION_PROMPT,
            "information_extraction": INFORMATION_EXTRACTION_PROMPT,
        }
        self.current_prompt_key = "cot_action" 
        self.model_name = model_name 

    def set_prompt(self, prompt_key):
        if prompt_key in self.prompts:
            self.current_prompt_key = prompt_key
        else:
            raise ValueError(f"Prompt key '{prompt_key}' not found. Available keys: {list(self.prompts.keys())}")

    def generate_action(self, observation, task_description, **kwargs):
        prompt_template = self.prompts[self.current_prompt_key] 
        
        prompt_inputs = {"observation": observation, "task_description": task_description} 
        prompt_inputs.update(kwargs) 
        
        prompt = prompt_template.format(**prompt_inputs)
        
        tqdm.write(f"{COLOR_MAGENTA}  [{self.model_name}] Invoking LLM for action generation...{COLOR_RESET}") 
        response = self.llm.invoke(prompt) 
        tqdm.write(f"{COLOR_MAGENTA}  [{self.model_name}] LLM invocation completed.{COLOR_RESET}") 
        
        thought = ""
        action_text = ""
        extracted_info = ""
        
        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith("Thought:"):
                thought = line[len("Thought:"):].strip()
            elif line.startswith("Action:"):
                action_text = line[len("Action:"):].strip()
            elif line.startswith("Extracted Information:"): 
                extracted_info = line[len("Extracted Information:"):].strip()

        if self.current_prompt_key == "information_extraction":
            return thought, extracted_info, action_text
        else:
            return thought, action_text

    def perform_android_action(self, simulated_env, action_dict):
        return simulated_env.step(action_dict)


def run_evaluation(llm_agent, simulated_env, task_description, prompt_key="cot_action", **prompt_kwargs):
    llm_agent.set_prompt(prompt_key) 

    observation = simulated_env.reset(task_description) 
    done = False
    steps = 0
    max_steps = 5
    
    reasoning_trace = [] 

    for step_num in tqdm(range(max_steps), desc=f"Task '{task_description}' steps", leave=False, colour="green"): 
        if done: 
            break

        processed_observation = process_android_observation(observation)
        
        if prompt_key == "navigation":
            thought, action_text = llm_agent.generate_action(processed_observation, task_description, navigation_goal=task_description)
        elif prompt_key == "information_extraction":
            thought, extracted_info, action_text = llm_agent.generate_action(processed_observation, task_description, query="Extract the total price.") 
        else:
            thought, action_text = llm_agent.generate_action(processed_observation, task_description)
        
        reasoning_trace.append({"step": steps + 1, "observation": processed_observation, "thought": thought, "action": action_text})

        android_action = parse_llm_action(action_text)
        
        observation, reward, done, info = llm_agent.perform_android_action(simulated_env, android_action)

        steps += 1

    success = info.get("success", False) 
    return {"success": success, "steps": steps, "reasoning_trace": reasoning_trace, "task_goal": task_description}


def process_android_observation(observation):
    ui_elements_text = ""
    for element in observation["ui_elements"]:
        ui_elements_text += f"{element['text']} (id: {element['resource_id']}), "
    return f"Current screen: {ui_elements_text}"


def parse_llm_action(action_text):
    if action_text.startswith("click("):
        element_id = action_text.split("id:").strip(")")
        return {"action_type": "click", "element_id": element_id}
    elif action_text.startswith("input("):
        parts = action_text.split(":")
        element_id_text_part = parts.strip(")")
        element_id = element_id_text_part.split(",").strip()
        text = element_id_text_part.split("text:").strip('"')
        return {"action_type": "input", "element_id": element_id, "text": text}
    elif action_text.startswith("scroll("):
        direction = action_text.split("direction:").strip(")")
        return {"action_type": "scroll", "direction": direction}
    return {"action_type": "no_op"}


if __name__ == "__main__":
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True) 

    for model_name in tqdm(MODELS_TO_COMPARE, desc="Evaluating Models", position=0, colour="blue"): 
        safe_model_name = model_name.replace("/", "__").replace("-", "_").lower()
        model_results_dir = os.path.join(BASE_RESULTS_DIR, safe_model_name)
        os.makedirs(model_results_dir, exist_ok=True)
        tqdm.write(f"\n{COLOR_BLUE}--- Starting evaluation for model: {model_name} ---{COLOR_RESET}") 

        simulated_env = SimulatedAndroidWorldEnv(task_data_path=TASK_FILE) 
        llm_agent = HFLLMAgent(model_name=model_name) 

        tasks_to_run = simulated_env.get_available_tasks() 

        all_task_results = [] 
        for task_id, task_info in tqdm(tasks_to_run.items(), desc=f"Tasks for {safe_model_name}", leave=False, position=1, colour="cyan"): 
            task_goal = task_info["goal"]
            prompt_key = task_info.get("prompt", "cot_action") 
            
            run_kwargs = {"prompt_key": prompt_key}
            if prompt_key == "navigation":
                run_kwargs["navigation_goal"] = task_info.get("navigation_goal", task_goal)
            elif prompt_key == "information_extraction":
                run_kwargs["query"] = task_info.get("query", "Extract information.")

            task_results = run_evaluation(llm_agent, simulated_env, task_goal, **run_kwargs)
            all_task_results.append(task_results) 

            filename = f"{task_id.replace(' ', '_').lower()}_results.json" 
            filepath = os.path.join(model_results_dir, filename) 
            tqdm.write(f"{COLOR_YELLOW}  Storing results for task '{task_goal}' in {filepath}...{COLOR_RESET}") 
            with open(filepath, "w") as f: 
                json.dump(task_results, f, indent=4) 
            tqdm.write(f"{COLOR_YELLOW}  Results saved for task '{task_goal}'.{COLOR_RESET}") 
            
            tqdm.write(f"{COLOR_GREEN}  Success: {task_results['success']}, Steps: {task_results['steps']}{COLOR_RESET}")
            
            tqdm.write(f"\n{COLOR_CYAN}  Reasoning Trace for '{task_goal}':{COLOR_RESET}")
            for entry in task_results["reasoning_trace"]:
                tqdm.write(f"    Step {entry['step']}:")
                tqdm.write(f"      Observation: {entry['observation']}")
                tqdm.write(f"      Thought: {entry['thought']}")
                tqdm.write(f"      Action: {entry['action']}")
            tqdm.write(f"{COLOR_RESET}" + "-" * 30) # Ensure reset at the end

        total_successes = sum(1 for r in all_task_results if r["success"])
        total_tasks = len(all_task_results)
        success_rate = (total_successes / total_tasks * 100) if total_tasks > 0 else 0 
        average_steps = sum(r["steps"] for r in all_task_results) / total_tasks if total_tasks > 0 else 0

        overall_results = {
            "model_name": model_name, 
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
            "total_tasks": total_tasks,
            "total_successes": total_successes,
            "success_rate": f"{success_rate:.2f}%",
            "average_steps": f"{average_steps:.2f}",
            "individual_task_results": all_task_results 
        }

        overall_filename = f"overall_evaluation_results_{safe_model_name}_{overall_results['timestamp']}.json"
        overall_filepath = os.path.join(model_results_dir, overall_filename) 
        tqdm.write(f"\n{COLOR_YELLOW}Storing overall results for model '{model_name}' in {overall_filepath}...{COLOR_RESET}") 
        with open(overall_filepath, "w") as f: 
            json.dump(overall_results, f, indent=4) 
        tqdm.write(f"{COLOR_YELLOW}Overall results for model '{model_name}' saved.{COLOR_RESET}")

        tqdm.write(f"\n{COLOR_BLUE}Overall Results for {model_name}:{COLOR_RESET}")
        tqdm.write(f"  Total tasks: {total_tasks}")
        tqdm.write(f"  Success rate: {overall_results['success_rate']}")
        tqdm.write(f"  Average steps: {average_steps:.2f}") 
        tqdm.write("=" * 50) 
