# prompts.py

from langchain_core.prompts import PromptTemplate

SUGGESTED_AGENT_PROMPT = PromptTemplate.from_template(
    """Goal: Uninstall the Slack app
    Observation:
    - App: Settings
    - UI Elements: ["Apps", "Search", "Battery", ...]
    What is the next best action? Respond in the format:
    CLICK("Apps")
    """
)

# Prompt for general Android agent action with Chain-of-Thought
ANDROID_AGENT_COT_PROMPT = PromptTemplate.from_template(
    """You are an Android agent. Your goal is to complete the given task by interacting with the Android app.
    You will be provided with the current screen's UI elements and the task description.
    Based on this information, first, **think step-by-step to plan the next action**, then generate the action.

    Current Screen:
    {observation}

    Task: {task_description}

    Your response should be in the format:
    Thought: [Your step-by-step reasoning]
    Action: [Your single, valid Android action in the format: click(id: element_id) or input(id: element_id, text: "your text") or scroll(direction: up/down/left/right)]
    """
)

# Prompt for navigation-specific tasks
NAVIGATION_PROMPT = PromptTemplate.from_template(
    """You are an Android navigation expert. Your task is to navigate to a specific screen based on the instructions.
    Analyze the current screen and determine the best action to reach the destination.

    Current Screen:
    {observation}

    Navigation Goal: {navigation_goal}

    Think step-by-step and provide the optimal action.
    Your response should be in the format:
    Thought: [Your navigation reasoning]
    Action: [Your single, valid Android action]
    """
)

# Prompt for information extraction tasks
INFORMATION_EXTRACTION_PROMPT = PromptTemplate.from_template(
    """You are an information extraction agent within an Android app.
    Your goal is to extract specific information from the current screen based on the query.

    Current Screen:
    {observation}

    Information Query: {query}

    Identify the relevant UI element and extract the required information.
    Your response should be in the format:
    Thought: [Your extraction reasoning]
    Extracted Information: [The extracted data, e.g., text, value]
    Action: [Optional, the action to take after extraction, or 'None']
    """
)
