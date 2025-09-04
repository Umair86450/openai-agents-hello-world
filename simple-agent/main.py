from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_default_openai_client, set_tracing_disabled
from dotenv import load_dotenv
import os

# Load environment variables (API key)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Create AsyncOpenAI client with the API key
external_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=api_key
)

# Set default OpenAI client
set_default_openai_client(external_client)
set_tracing_disabled(True)

# Model setup (You can use gemini-2.5-flash or any other model)
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

# Function to generate study questions and answers for a given topic
def run_study_assistant(user_input):
    """
    Function to generate questions and answers for a given topic related to climate change in Pakistan.
    
    Args:
    user_input (str): The topic or subject for which questions and answers are generated.

    Returns:
    str: The assistant's response (questions and answers).
    """
    # Define the AI Study Assistant Bot for Climate Change in Pakistan
    agent = Agent(
        name="ClimateChangeAssistantBot",
        instructions="""
            You are an AI assistant focused on climate change issues in Pakistan. Given a topic, 
            generate relevant questions about climate change in Pakistan, and provide clear and concise answers to each question.
            Example:
            - For the topic 'Impact of Climate Change on Pakistan', generate questions like:
              1. What are the key impacts of climate change in Pakistan?
              2. How does climate change affect agriculture in Pakistan?
              3. What is the role of water scarcity in climate change in Pakistan?
            
            - For the topic 'Climate Change Adaptation Strategies in Pakistan', generate questions like:
              1. What strategies can Pakistan implement to adapt to climate change?
              2. How can Pakistan improve water management to fight climate change?
              3. What measures can be taken to promote climate-resilient agriculture in Pakistan?

            Provide clear and insightful answers to each of these questions, focusing on actionable steps and solutions that Pakistan can take to mitigate climate change effects.
        """,
        model=model
    )

    # Create the prompt dynamically based on the user input (topic)
    prompt = f"Generate 2 relevant questions related to the topic 'Climate Change in Pakistan'. Then provide clear and concise answers to each question. Make sure the questions focus on issues like climate adaptation strategies, water scarcity, and agriculture in Pakistan."

    # Run the dynamic prompt through the assistant
    result = Runner.run_sync(agent, prompt)

    # Return the assistant's response
    return result.final_output

# Function to handle the interactive loop with the user
def start_study_assistant():
    """
    Starts the interactive loop where the user provides topics, 
    and the assistant generates and answers related questions.
    """
    while True:
        user_input = input("You: Enter a topic (Climate Change, Water Scarcity, Agriculture) for study assistance: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        # Call the function to get the questions and answers for the given topic
        response = run_study_assistant(user_input)
        
        # Display the assistant's response
        print("Assistant: " + response)

# Start the assistant
start_study_assistant()
