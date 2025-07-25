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

A First Example of a Failed Task:

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

A Second Example of a Failed Task:

```
Goal: "Open the file task.html in Downloads in the file manager; when prompted open it with Chrome. Then click the button 5 times, remember the numbers displayed, and enter their product in the form."
----------step 1
Action: {"action_type": "navigate_home"}
Reason: To begin the task of opening the file "task.html" in Downloads, we first need to access the file manager on the Android device. Since the file manager app is not directly visible on the current home screen, we will open the app drawer to locate and open the file manager.
Summary: Navigated to the home screen to access the app drawer for locating the file manager, as it was not directly visible on the initial screen. The action was successful, and the home screen is now displayed, ready for the next step to open the app drawer.
Completed step 1.
----------step 2
Action: {"action_type": "click", "index": 0}
Reason: The user's goal is to open the file "task.html" in Downloads using the file manager and then open it with Chrome. Currently, the file manager app is not visible on the home screen. To proceed, I need to access the app drawer where additional apps, including the file manager, might be located.
Summary: Attempted to access the app drawer by clicking on the screen, but the action did not change the screen state, indicating the click may have been ineffective or misdirected. Need to reassess the approach to open the app drawer.
Completed step 2.
----------step 3
Action: {"action_type": "click", "index": 11}
Reason: The user's goal is to open the file "task.html" in Downloads using the file manager and then use Chrome to view it. Since the file manager app is not directly visible on the home screen, the next step is to open the app drawer to locate the file manager. The previous attempt to open the app drawer by clicking index 0 was ineffective. Observing the current screenshot, the Google app icon at index 11 is typically associated with a search or app suggestion functionality on Android devices, which might help in locating or opening the file manager directly.
Summary: Clicked on the Google app icon (index 11) to access search or app suggestion functionality, aiming to locate the file manager. This action redirected to the Google search page instead of opening the app drawer or locating the file manager. Need to find another way to access the file manager.
Completed step 3.
----------step 4
Action: {"action_type": "input_text", "text": "file manager", "index": 1}
Reason: Since the previous attempts to access the app drawer directly from the home screen were unsuccessful, and the Google app did not provide a direct route to the file manager, the next step is to use the search functionality to directly search for and open the file manager app. This can be done by typing "file manager" into the Google search bar.
Summary: Typed "file manager" into the Google search bar to locate and open the file manager app directly, resulting in a search results page with various file manager options and related queries.
Completed step 4.
----------step 5
Action: {"action_type": "click", "index": 35}
Reason: The search results for "file manager" have been displayed, and there are options to install various file manager apps. To proceed with the user's goal of opening a file in the file manager, I will choose to install the "Files by Google" app, which is a well-known and reliable file manager. The clickable element to install this app is indexed at 35.
Summary: Clicked on "Install Files by Google" to download a file manager app, but received an error message "No app found to open link," indicating the action failed to initiate the app installation. Need to explore alternative methods to access or install a file manager.
Completed step 5.
----------step 6
Action: {"action_type": "click", "index": 50}
Reason: The previous attempts to install a file manager from the search results were unsuccessful. To proceed with the user's request to open a file in a file manager, it's necessary to try a different approach. Since the search results include clickable options for installing file managers, I will attempt to install the "Files by Google" app, which is a reliable and commonly used file manager on Android devices.
Summary: Attempted to install "Files by Google" from the search results to access a file manager, but received an error message "No app found to open link," indicating the action did not succeed. Need to explore alternative methods to access or install a file manager.
Completed step 6.
----------step 7
Action: {"action_type": "open_app", "app_name": "Files by Google"}
Reason: The previous attempts to install a file manager from the Google search results were unsuccessful. To proceed with the user's goal of opening a file in a file manager, I will attempt to open an app directly using the "open_app" action. This action will attempt to open the "Files by Google" app, which is commonly used for file management on Android devices.
Summary: Successfully opened "Files by Google" app, navigated to the Downloads folder, and located the file "task.html". Ready for the next action to open the file with Chrome.
Completed step 7.
----------step 8
Action: {"action_type": "long_press", "index": 14}
Reason: The next step in the user's goal is to open the file "task.html" located in the Downloads folder. The file is visible on the screen, but it is not clickable according to the UI element details. To proceed, I will attempt to long press the file, which might trigger a context menu or another method to open the file.
Summary: Long-pressed "task.html" to trigger selection; file is now selected, indicating potential next steps to open or manage the file.
Completed step 8.
----------step 9
Action: {"action_type": "click", "index": 4}
Reason: The file "task.html" is currently selected, and the next step is to open it. Since the user's goal specifies opening the file with Chrome, I will click on the "More options" button (index 4) to access additional options, which likely includes an "Open with" choice where Chrome can be selected.
Summary: Clicked on "More options" to access file operations for "task.html"; "Open with" option now visible, ready to select Chrome for opening the file.
Completed step 9.
----------step 10
Action: {"action_type": "click", "index": 0}
Reason: The user's goal is to open the file "task.html" with Chrome. The "Open with" option is visible on the screen, which is the next step to select Chrome as the application to open the file.
Summary: Clicked on "Open with" to choose an application for opening "task.html"; screen updated to show app options including Chrome, but Chrome is not clickable yet.
Completed step 10.
----------step 11
Action: {"action_type": "scroll", "direction": "down"}
Reason: The current screen shows the app selection interface with Chrome and HTML Viewer as options to open the file "task.html". However, the Chrome app is not clickable at the moment. To proceed with opening the file in Chrome as per the user's request, I need to scroll down to potentially reveal more options or make the Chrome app clickable.
Summary: Scrolled down to make Chrome clickable for opening "task.html"; Chrome remains unclickable, further action needed to enable selection.
Completed step 11.
----------step 12
Action: {"action_type": "scroll", "direction": "down"}
Reason: The previous attempt to scroll down did not reveal Chrome as a clickable option for opening the file. To ensure that Chrome becomes visible and selectable, another scroll action is necessary. This will potentially move the screen further down to make Chrome accessible.
Summary: Performed a second scroll down to reveal more options for opening the file, aiming to find and select Chrome. The screen moved further down, but Chrome is still not visible. Further scrolling or a different approach may be needed.
Completed step 12.
----------step 13
Action: {"action_type": "scroll", "direction": "up"}
Reason: After multiple attempts to scroll and reveal Chrome as an option to open the file "task.html" and still not being able to see it, it seems necessary to try a different approach. Since the previous actions to scroll down did not reveal Chrome, I will try scrolling up in case the list of apps was overscrolled or Chrome is located higher in the list.
Summary: Attempted to scroll up to locate Chrome for opening "task.html" but still did not reveal the app; may need to explore alternative methods to access Chrome or check app availability.
Completed step 13.
----------step 14
Action: {"action_type": "scroll", "direction": "down"}
Reason: The previous attempts to scroll and locate Chrome for opening the file "task.html" were unsuccessful. To proceed with the task, I will attempt to scroll down again to reveal more options, hoping to find and select Chrome.
Summary: Attempted to scroll down to locate Chrome for opening "task.html" but the screen did not reveal new options; the action did not change the visible options. Further scrolling or a different approach may be needed to find Chrome.
Completed step 14.
----------step 15
Action: {"action_type": "navigate_back"}
Reason: After multiple attempts to scroll and locate Chrome for opening the file "task.html" without success, it seems that Chrome might not be available or not functioning as expected in the current context. To proceed with the user's goal of opening the file in Chrome, an alternative approach is needed. Since direct selection of Chrome has been unsuccessful, I will attempt to navigate back and reassess the situation or find another way to open the file with Chrome.
Summary: Navigated back to reassess the situation after unsuccessful attempts to open "task.html" with Chrome due to its apparent unavailability. Need to explore alternative methods to open the file.
Completed step 15.
----------step 16
Action: {"action_type": "click", "index": 9}
Reason: After navigating back to reassess the situation, the screen now shows the "Downloads" folder with the file "task.html" visible. To proceed with opening the file with Chrome, the next step is to click on the file to bring up the options for opening it.
Summary: Clicked on "task.html" to open it, successfully prompted with app options for opening the file. Next, need to select Chrome to proceed with opening the file.
Completed step 16.
----------step 17
Action: {"action_type": "scroll", "direction": "down"}
Reason: The Chrome app is currently not clickable, which is necessary to proceed with opening the file "task.html" as per the user's request. To make Chrome clickable, I will attempt to scroll down again to possibly reveal more options or make the Chrome option selectable.
Summary: Attempted to make the Chrome app selectable by scrolling down, but the screen did not change significantly, and Chrome remains unclickable. Further action may be needed to access Chrome or adjust the view.
Completed step 17.
----------step 18
Action: {"action_type": "navigate_back"}
Reason: After several attempts to scroll and make Chrome visible and clickable for opening the file "task.html", it appears that Chrome is not available or not being displayed correctly in the list of apps to open the file. To proceed with the task, I will attempt to navigate back to reassess the situation and explore alternative methods to open the file with Chrome.
Summary: Navigated back due to inability to find Chrome for opening "task.html"; reassessing options to proceed with the task.
Completed step 18.
----------step 19
Action: {"action_type": "scroll", "direction": "down"}
Reason: The user's goal is to open the file "task.html" in Downloads using Chrome. The previous attempts to open the file with Chrome were unsuccessful due to the inability to select Chrome from the app options. To proceed, I will attempt to scroll down again to reveal Chrome as a selectable option for opening the file.
Summary: Scrolled down to reveal more options for file managers in the search results, but still did not find Chrome or the desired file manager app to proceed with opening "task.html". Further action may be needed to locate and install a suitable file manager.
Completed step 19.
----------step 20
Action: {"action_type": "scroll", "direction": "down"}
Reason: The user's goal is to open the file "task.html" in Downloads using Chrome. The file has been located in the "Files by Google" app, and the next step is to select Chrome to open the file. From the history, it appears that Chrome was not visible or clickable in the app selection menu. To proceed, I will attempt to scroll down again to reveal Chrome as an option for opening the file.
Summary: Scrolled down to reveal more options for file managers in the search results, but Chrome still not visible for selection. Need to continue scrolling or find another way to access Chrome.
Completed step 20.
----------step 21
Action: {"action_type": "click", "index": 53}
Reason: The current screen shows search results for file managers on Google, and there is a clickable link for "File Manager – Apps on Google Play" which seems to be a direct link to the app on the Google Play Store. Clicking this link should take us to the Google Play Store where we can install the app.
Summary: Clicked on the "File Manager – Apps on Google Play" link in search results, successfully redirected to the Google Play Store page for the app in Chrome, now on the welcome screen of Chrome.
Completed step 21.
----------step 22
Action: {"action_type": "click", "index": 0}
Reason: To proceed with using Chrome to open the file "task.html", it is necessary to accept the terms and conditions presented on the welcome screen of Chrome. The "Accept & continue" button is clickable and needs to be activated to move forward.
Summary: Clicked "Accept & continue" in Chrome to proceed with the setup, successfully moved to the next screen prompting "Turn on sync?". Next action should address this new prompt.
Completed step 22.
Agent did not indicate task is done. Reached max number of steps.
Task Failed ❌;  Open the file task.html in Downloads in the file manager; when prompted open it with Chrome. Then click the button 5 times, remember the numbers displayed, and enter their product in the form.
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

| Task                                 | Max steps | From GitHub |  My run | Reduction |
| ------------------------------------ | --------- | ----------- | ------- | --------- |
| AudioRecorderRecordAudioWithFileName | 20        | 12          | 15      | \-3       |
| BrowserDraw                          | 20        | 20          | 20      | 0         |
| BrowserMaze                          | 20        | 20          | 20      | 0         |
| BrowserMultiply                      | 22        | 22          | 22      | 0         |
| SimpleSmsReply                       | 12        | 12          | 5       | 7         |
| ContactsAddContact                   | 12        | 12          | 8       | 4         |


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

- `distilbert-base-uncased`: DistilBERT is a distilled version of BERT (Bidirectional Encoder Representations from Transformers), meaning it's a smaller, faster version trained to mimic the behavior of the larger BERT model. BERT revolutionized NLP by understanding context from both left-to-right and right-to-left in a sentence. DistilBERT retains about 97% of BERT's performance while being 40% smaller and 60% faster. It is well-suited for tasks like text classification, sentiment analysis, named entity recognition, and question answering.

- `gpt2` : GPT-2 (Generative Pre-trained Transformer 2) is a Transformer architecture, specifically a decoder-only model. It was known for its size (up to 1.5 billion parameters in its largest version) on release.

# Run Analysis 

The prediction using the models above takes significantly longer (one prediction using `HuggingFaceH4/zephyr-7b-beta` takes 2383.94 seconds more than the GPT 4.1) so I did not run with all of them but the set up is ready to run with them. 

The following table shows the completion time of a prediction for various models:

| Model                              | Completion Time |
| ---------------------------------- | --------------- |
| distilbert-base-uncased            | 0.47            |
| gpt2                               | 10.39           |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.17            |
| google/gemma-2b-it                 | No access       |
| HuggingFaceH4/zephyr-7b-beta       | \> 1 min        |
| stabilityai/stablelm-2-zephyr-1_6b | 48.91           |
| mistralai/Mistral-7B-v0.1          | No access       |
| claude-opus-4-20250514             | No access       |
| microsoft/Phi-3-mini-4k-instruct   | \> 1 min        |

Given the time it takes for each of the model to predict in one step we decided to use for comparison the models that are faster and less than 1 minute. Out of the shorted list the `distilbert-base-uncased` is optimized for classifications and sentiment analysis but not for the text generation. `TinyLlama/TinyLlama-1.1B-Chat-v1.0` model failed to generate meaningful text according with the expected answer. The GPT 2 failed to answer the answer successfully and is a model inferior to GPT 4.1. Lastly, the `stabilityai/stablelm-2-zephyr-1_6b` also failed to answer as expected.

## Considerations for improvement

- Prompt Engineering: Optimize the prompts provided to the LLM for better instruction following and reasoning.
- Advanced Observation Processing: Experiment with methods like generating summaries, visual grounding, or incorporating historical context to improve the LLM's understanding of the UI.
- Complex Action Parsing: Support a wider range of Android actions and complex action sequences.
- Human Evaluation: Incorporate human judgment for subjective assessment and to validate the LLM's performance on intricate tasks.

