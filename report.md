# Lightweight evaluation framework for LLMs as mobile agents in AndroidWorld

This **QualGent LLM evaluation framework** focuses on evaluating the ability of LLMs to act as agents in simulated Android environments using the AndroidWorld benchmark. It assesses how well LLMs can interpret observations, generate valid actions, and achieve specific goals within simulated Android apps.

To elaborate on evaluating planning and reasoning within this framework, the focus needs to shift beyond just task completion to analyzing the process through which the LLM agent reaches its goal. LLMs might solve problems through pattern matching rather than genuine reasoning, so it's important to differentiate between those approaches. 
This a way to evaluate planning and reasoning:

## Components

- AndroidWorld Environment: Utilizes the AndroidWorld benchmark for realistic Android app simulations and tasks.
- LLM Agent: The LLM being evaluated, receiving observations and generating actions.
- Observation Processing: Transforms AndroidWorld's observations (e.g., UI elements, text descriptions, screenshots) into a format suitable for the LLM.
- Action Generation: Enables the LLM to generate actions (e.g., clicks, scrolls, text inputs) based on its reasoning.
- Execution and Feedback: Applies the LLM's generated actions within the AndroidWorld environment and provides updated observations to the LLM.
- Evaluation Metrics: Quantifies the LLM's performance based on task success, efficiency, and robustness.

## Key metrics for evaluation

- Success Rate: Percentage of tasks successfully completed by the LLM agent.
- Steps to Completion/Minimum episode lenght: Number of actions taken to complete a task, indicating efficiency. This metric becomes more insightful when analyzing unexpected detours or inefficient paths, suggesting potential reasoning flaws.
- Robustness to variations: How well the agent handles variations, errors, or unexpected UI elements. Test the LLM's performance under slightly altered conditions, such as rephrased instructions or minor UI changes. Consistent success across variations indicates stronger reasoning and generalization capabilities, suggesting it's not merely relying on memorized patterns. 
- Planning and Multi-Step Reasoning: Assessed by the complexity of tasks that the agent can successfully navigate. In my_agent class, I modified the PROMPT_PREFIX to include planning and step-by-ste analysis in the LLM model. The model responded well and shows it was using this recommendations.
- Action Validity and Sequencing: The percentage of generated actions that are valid and executable within the Android environment. Beyond simply generating valid actions, evaluate whether the actions form a coherent and logical sequence to achieve the task goal. For example, using the calculator to perform a calculation before navigating to the settings menu for brightness adjustments would indicate a lack of proper planning.
- Deviation from optimal plan: Define an optimal sequence of actions for each task, allowing comparison with the agent's actual steps. This highlights situations where the LLM deviates from efficient solutions, potentially revealing weaknesses in its planning capability.
- Task decomposition and sub-goal attainment: For complex tasks, define intermediate sub-goals. Evaluating the agent's ability to identify and successfully complete these sub-goals provides a more nuanced understanding of its reasoning and planning abilities.

## Techniques

- Prompt engineering and analysis:
Chain-of-Thought (CoT) prompting: Encourage the LLM to explicitly articulate its reasoning steps in addition to generating actions. This can be achieved by adding prompts like "Explain your thought process before generating an action." or "Based on the screen and task, first think step-by-step to plan the next action, then generate the action". Analyzing the quality of these intermediate thoughts can provide insights into the LLM's reasoning process.

- Few-shot prompting: Provide examples of successful reasoning and action sequences to guide the LLM toward desired planning behaviors.

- Error analysis: Delve deeper into cases where the agent fails or takes suboptimal paths. This involves analyzing the observation history, generated actions, and reasoning steps (if CoT is used) to pinpoint where the LLM's understanding or planning went wrong.

- Comparing LLM performance: Test different LLMs with varying sizes and architectures on the same tasks to observe how reasoning and planning capabilities evolve.
Integrating external critics or verifiers: Consider incorporating a system to evaluate the LLM's proposed plan before execution, identifying potential errors or suboptimal steps.

## Initial minimal task runner with GPT 4.1

An initial example of a successful task:

```
Goal: Create a new contact for Ian Gonzalez. Their number is +12477368853.
----------step 1
Action: {"action_type": "open_app", "app_name": "Contacts"}
Reason: To create a new contact for Ian Gonzalez with the provided phone number, I need to access the Contacts app. Since the Contacts app is not directly visible on the current screen, I will use the "open_app" action to open it directly.
Summary: Opened the Contacts app to add a new contact for Ian Gonzalez, but encountered a permissions prompt asking to allow Contacts to send notifications. Need to decide on permission to proceed with contact creation.
----------step 2
Action: {"action_type": "click", "index": 1}
Reason: To proceed with creating a new contact, the Contacts app needs permission to send notifications. The prompt on the screen is asking whether to allow this permission. Granting this permission is necessary to continue with the task.
Summary: Clicked 'Allow' to grant the Contacts app permission to send notifications, successfully transitioning to the Contacts app main screen with an option to create a new contact visible. This action was necessary to proceed with adding a new contact as per the user's request.
----------step 3
Action: {"action_type": "click", "index": 1}
Reason: To proceed with creating a new contact for Ian Gonzalez, the next step is to initiate the contact creation process by clicking on the "Create contact" button, which is represented by UI element 1.
Summary: Clicked on the "Create contact" button to initiate the process of adding a new contact for Ian Gonzalez. The screen transitioned to the contact creation form as expected, displaying fields for entering contact details such as name, phone number, and email.
----------step 4
Action: {"action_type": "input_text", "text": "Ian", "index": 7}
Reason: The next step in creating a new contact for Ian Gonzalez involves entering the first name "Ian" into the appropriate text field. The UI element index for the "First name" field is 7, which is editable and enabled.
Summary: Entered the first name "Ian" into the 'First name' field successfully, preparing to input the last name next.
----------step 5
Action: {"action_type": "input_text", "text": "Gonzalez", "index": 8}
Reason: The next step in creating a new contact for Ian Gonzalez is to enter his last name, "Gonzalez", into the 'Last name' field. The UI element index for the 'Last name' field is 8, which is editable and visible on the screen.
Summary: Entered the last name "Gonzalez" into the 'Last name' field successfully, preparing to input additional contact details next.
----------step 6
Action: {"action_type": "input_text", "text": "+12477368853", "index": 10}
Reason: The next step in creating a new contact for Ian Gonzalez is to input his phone number. From the previous actions, the first name and last name have been successfully entered. The UI element for entering the phone number is identified as index 10.
Summary: Entered the phone number "+12477368853" for the new contact Ian Gonzalez into the phone number field (UI element index 10), successfully updating the contact details as intended. This action is part of the process to create a new contact in the Contacts app.
----------step 7
Action: {"action_type": "click", "index": 2}
Reason: All necessary information for creating the new contact for Ian Gonzalez has been entered, including first name, last name, and phone number. The next logical step is to save the contact by clicking the 'Save' button.
Summary: Clicked the 'Save' button to store the new contact details for Ian Gonzalez after verifying that all necessary information (first name, last name, and phone number) was correctly entered. The screen transitioned to the contact list, indicating the contact was likely saved successfully.
----------step 8
Action: {"action_type": "status", "goal_status": "complete"}
Reason: The user's goal was to create a new contact for Ian Gonzalez with the specified phone number, and the previous steps have successfully completed this task. The contact has been saved, and the screen now displays the contact details, confirming that the contact has been added to the list. Therefore, the task is complete.
Task Successful ✅; Create a new contact for Ian Gonzalez. Their number is +12477368853.
```

An initial example of a Failed Task:

```
Goal: Reply to +15910808480 with message: Inconceivable! in Simple SMS Messenger
----------step 1
Action: {"action_type": "open_app", "app_name": "Simple SMS Messenger"}
Reason: The user's goal is to reply to a specific phone number using the Simple SMS Messenger app. The current screen is the home screen of the Android device, and the Simple SMS Messenger app is not directly visible among the listed apps. Therefore, the first step is to open the Simple SMS Messenger app to proceed with sending the message.
Summary: Opened the Simple SMS Messenger app to enable replying to the specified phone number as per the user's request. The home screen did not show the app directly, necessitating the use of the 'open_app' action. The next steps will involve locating the conversation with the specific number and sending the message.
----------step 2
Action: {"action_type": "open_app", "app_name": "Simple SMS Messenger"}
Reason: The previous action attempted to open the Simple SMS Messenger app, but it seems that the action led to the home screen instead of opening the app. This could be due to the app not being installed or an error in opening the app. To proceed with the task of sending a message, I will attempt to open the app again to ensure it is indeed installed and accessible.
Summary: Attempted to open Simple SMS Messenger to send a message, but the action returned to the home screen, possibly indicating the app is not installed or there was an error in launching it. Need to verify app installation or consider alternative messaging apps.
----------step 3
Action: {"action_type": "click", "index": 3}
Reason: Since the attempt to open the Simple SMS Messenger app failed twice, indicating that the app might not be installed or there is an issue with launching it, an alternative approach is needed to send the SMS. The default Messages app, which is typically pre-installed on Android devices, can be used to send the SMS. UI element 3 is labeled as 'Messages' and is clickable, making it a suitable choice to proceed with sending the message.
Summary: Clicked on 'Messages' app to attempt sending SMS through the default messaging application after failing to open Simple SMS Messenger, successfully navigated to the Messages app interface.
----------step 4
Action: {"action_type": "scroll", "direction": "down"}
Reason: The user's goal is to reply to the phone number +15910808480 with the message "Inconceivable!" using the Messages app. From the provided UI elements, none of the visible conversation names match the target phone number. Therefore, the next step is to scroll down the conversation list to search for the conversation with the specified phone number.
Summary: Scrolled down the conversation list to locate the conversation with +15910808480, but the target number was not found in the visible list; further scrolling or searching may be required to find the correct conversation.
----------step 5
Action: {"action_type": "status", "goal_status": "infeasible"}
Reason: The target phone number +15910808480 is not found in the visible list of conversations. The closest match in the list is (591) 080-8480, which is not the correct number. Since the correct conversation cannot be located and the Simple SMS Messenger app could not be opened, it is not feasible to send the message as requested.
Agent stopped since it thinks mission impossible.
Task Failed ❌; Reply to +15910808480 with message: Inconceivable! in Simple SMS Messenger
```

## Test suite run (only 5 tasks) with GPT 4.1 

| task                                 | task_num | num_complete_trials | mean_success_rate | mean_episode_length | total_runtime_s | num_fail_trials |
| ------------------------------------ | -------- | ------------------- | ----------------- | ------------------- | --------------- | --------------- |
| AudioRecorderRecordAudio             | 0        | 0                   | NaN               | NaN                 | 0.5             | 1               |
| AudioRecorderRecordAudioWithFileName | 1        | 1                   | 0                 | 15                  | 458             | 0               |
| BrowserDraw                          | 2        | 1                   | 0                 | 20                  | 445.1           | 0               |
| BrowserMaze                          | 3        | 1                   | 0                 | 20                  | 509.1           | 0               |
| BrowserMultiply                      | 4        | 1                   | 0                 | 22                  | 627.1           | 0               |
| \========= Average =========         | 0        | 0.8                 | 0                 | 19.25               | 407.96          | 0.2             |

The reasons each task failed was:

* Task 1: NaN
* Task 2: The agent failed to open the Audio Recorder app.
* Task 3: The agent failed to open the File Manager app. 
* Task 4: The agent failed to click in the Chrome option to open the task.html file.
* Task 5: The agent failed to find the Chrome option clickable.

The percentage of valid actions taken in each task is (task 6 is task 0 in the table and the other tasks have same order):

<img src="./images/Picture1.png" width="400" height="200" alt="The Percentage of Valid actions taken by the agent">


The complexity of the tasks using difficulty tags:

| Difficulty Tags          | Easy | Hard  |
| ------------------------ | -----| ----- |
| Complex_UI_Understanding | ✅   | ❌     |
| Data_Entry               | ✅   | ❌     |
| Game_Playing             | ✅   | ❌     |
| Math_Counting            | ❌   | ✅     |
| Memorization             | ✅   | ❌     |
| Multi_App                | ✅   | ❌     |
| Parameterized            | ✅   | ❌     |
| Screen_Reading           | ❌   | ✅     |

## Task Decomposition and Planned Tasks by the LLM agent

![Alt text](./images/tasks_decomposition.drawio.svg "The Task Decomposition and the Planned Tasks by the LLM agent")

## Example of actions taken in a successful task

![Alt text](./images/AddContact.drawio.svg "Example of actions in a successful task")

## The Reduction in Steps to Completion/Episode Lenght

<style>
    .heatMap {
        width: 70%;
        text-align: center;
    }
    .heatMap th {
        background: black;
        word-wrap: break-word;
        text-align: center;
    }
    .heatMap tr:nth-child(1) { background: red; }
    .heatMap tr:nth-child(2) { background: grey; }
    .heatMap tr:nth-child(3) { background: grey; }
    .heatMap tr:nth-child(4) { background: grey; }
    .heatMap tr:nth-child(5) { background: red; }
    .heatMap tr:nth-child(6) { background: green; }
</style>

<div class="heatMap">
 

| Task                                 | From GitHub |  My run | Reduction |
| ------------------------------------ | ----------- | ------- | --------- |
| AudioRecorderRecordAudioWithFileName | 12          | 15      | \-3       |
| BrowserDraw                          | 20          | 20      | 0         |
| BrowserMaze                          | 20          | 20      | 0         |
| BrowserMultiply                      | 22          | 22      | 0         |
| SimpleSmsReply                       | 12          | 5       | 7         |
| ContactsAddContact                   | 12          | 8       | 4         |

</div>

## Simulated AndroidWorld benchmark

The need to avoid using the step function in the AndroidWorld benchmark presents a significant challenge, as the env.step(action) method is central to how reinforcement learning environments like AndroidWorld operate. It's the standard way an agent interacts with the environment: the agent provides an action, and the environment returns the next observation, reward, and whether the episode is finished. Additionally to the evaluation using the Android World environment, I decided to simulate this function due to lack of resources available to run with a GPT 4.1 model.

Potential reasons for avoiding env.step():

Direct manipulation: The user wants to directly interact with the Android environment without the abstract layer provided by env.step(). However, the documentation for AndroidWorld suggests using the step() function for interacting with the environment.
Experimenting with different execution mechanisms: Instead of relying on the environment to handle action execution, the user might want to implement a custom action execution mechanism within the agent or evaluation loop, bypassing the standard environment interaction. 

Challenges and limitations:

The env.step() method in reinforcement learning environments is designed to encapsulate the environment's dynamics, providing a standardized way to interact with it.
Without env.step(), the structured output of observation, reward, done status, and info is lost, which are crucial for evaluating and training agents.
These aspects would need to be implemented manually within the agent or evaluation loop, requiring a deeper understanding of the AndroidWorld environment's internal workings.

Possible approaches to avoid env.step() (if the user has a way to bypass it):

Since the env.step() function is fundamental to the AndroidWorld benchmark, if the user explicitly wants to avoid it, this implies that a different method or API might be used for interacting with the Android environment, or the interaction is being simulated outside the standard framework. 

## Directory structure

```
QualGent_LLM_evaluation/
├── main_evaluation_progress.py
├── android_world/
│   └── android_world/
│       └──run_modified.py
│       └── agents/
│           └── my_agent.py
├── prompts/
│   └── prompts.py
│   └── __init__.py  # Make 'prompts' a Python package
└── src/
    └── simulated_env.py
    └── __init__.py  # Make 'src' a Python package
```


## Creating my own agent

Defines a class `MyAndroidAgent` initializer for a custom agent, RandomAgent, which inherits from a superclass and sets up several instance variables, including an LLM wrapper `HFLLMAgent`, history, and step counters.

The constructor takes: an asynchronous environment (env), a language model wrapper (llm) and an optional agent name (default: 'my-agent'). It calls the superclass initializer with the environment and name. Sets up: self.llm to the provided LLM wrapper, self.history as an empty list, and self.additional_guidelines as None.

The agent uses the same step function as t3a but has modifications in the PROMPT_PREFIX to 
encourage self planning in the agent and step-by-step analysis and it add supports for multiple types of models.

The agent uses the suggested ACTION SELECTION PROMPT TEMPLATE:

```
Goal: Uninstall the Slack app
Observation:
- App: Settings
- UI Elements: ["Apps", "Search", "Battery", ...]
What is the next best action? Respond in the format:
CLICK("Apps")
```

We didn't observe any difference in terms of actions and steps to completion when modified the ACTION SELECTION PROMPT TEMPLATE so we didn't try with other templates.

## Dependencies 

```
pip install langchain_community
pip install transformers
pip install torch
pip install accelerate
```

# Usage 

```
python run_modified.py --agent_name=my_agent --model=HuggingFaceH4/zephyr-7b-beta --tasks=ContactsAddContact
```

## Models

Explanation of added models in the simulation framework and my_agent:

- `HuggingFaceH4/zephyr-7b-beta`: This is a 7B parameter, instruction-tuned model, fine-tuned from the Mistral-7B model using Direct Preference Optimization (DPO) on synthetic datasets. It's designed to act as a helpful assistant and excels at conversational dialogue, text generation, and tasks requiring reasoning and problem-solving. Zephyr 7B beta demonstrates strong performance on benchmarks like MT-Bench, even outperforming larger models like Llama2-Chat-70B in some areas. However, it may struggle with more complex tasks like coding or advanced mathematics compared to proprietary models.

- `mistralai/Mistral-7B-v0.1`: This is a 7.3 billion parameter base language model known for its efficiency and strong performance for its size. It incorporates advanced architectural features like Grouped-Query Attention and Sliding-Window Attention to handle longer text sequences more efficiently. It's a powerful generative text model, outperforming larger models like Llama 2 13B on various benchmarks. Mistral 7B is versatile and can be used for content generation, summarization, classification, and code completion. It is also designed for easy fine-tuning for specific tasks. This is a gated model and I did not use it.

- `microsoft/Phi-3-mini-4k-instruct`: This model from Microsoft is a small but capable instruction-tuned model. It's known for strong performance relative to its size, making it a good candidate for CPU-only inference with quantization.

- `google/gemma-2b-it`: Google DeepMind's Gemma 2B is another efficient model specifically designed to be lightweight and fast, suitable for resource-constrained environments.

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`: As the name suggests, TinyLlama is a very small model, ideal for quick testing and situations where minimal memory and computation are paramount. While its performance may not match the larger models, its speed on CPU is a definite advantage. 

- `stabilityai/stablelm-2-zephyr-1_6b`: This is a very small, 1.6 billion parameter model from Stability AI, trained on 2 trillion tokens. It's designed to be efficient and perform well in resource-constrained environments.

# Run Analysis 

The prediction using the models above takes significantly longer (one prediction using `HuggingFaceH4/zephyr-7b-beta` takes 2791.9 seconds) than the GPT 4.1 so I did not run with all of them but the set up is ready to run with them. There are more experiments still running.  

## Considerations for improvement

- Model agnostic: Run with the remaining LLM models. 
- Prompt Engineering: Optimize the prompts provided to the LLM for better instruction following and reasoning.
- Advanced Observation Processing: Experiment with methods like generating summaries, visual grounding, or incorporating historical context to improve the LLM's understanding of the UI.
- Complex Action Parsing: Support a wider range of Android actions and complex action sequences.
- Human Evaluation: Incorporate human judgment for subjective assessment and to validate the LLM's performance on intricate tasks.

