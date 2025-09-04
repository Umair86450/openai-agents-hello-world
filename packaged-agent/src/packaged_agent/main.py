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
    Function to generate questions and answers for a given topic.
    
    Args:
    user_input (str): The topic or subject for which questions and answers are generated.

    Returns:
    str: The assistant's response (questions and answers).
    """
    # Define the AI Study Assistant Bot for Agentic AI, Artificial Intelligence, and Autonomous Agents
    agent = Agent(
        name="AIStudyAssistantBot",
        instructions="""
            You are an AI assistant focused on topics related to Agentic AI, Artificial Intelligence, and Autonomous Agents. 
            Given a topic, generate several questions about this subject and provide clear and concise answers to each question. 
            Example:
            - For the topic 'Agentic AI', generate questions like:
              1. What is Agentic AI?
              2. How does Agentic AI differ from traditional AI?
              3. What are the applications of Agentic AI in autonomous systems?
            
            - For the topic 'Artificial Intelligence', generate questions like:
              1. What is Artificial Intelligence?
              2. What are the types of AI (e.g., Narrow AI, General AI)?
              3. How does machine learning relate to AI?

            - For the topic 'Autonomous Agents', generate questions like:
              1. What is an Autonomous Agent?
              2. How do Autonomous Agents function in a given environment?
              3. What are the ethical considerations of Autonomous Agents in society?

            Provide clear and insightful answers to each of these questions, making sure they are informative and accessible for someone new to these topics.
        """,
        model=model
    )

    # Create the prompt dynamically based on the user input (topic)
    prompt = f"Generate 3 relevant questions related to the topic '{user_input}' (e.g., Agentic AI, Artificial Intelligence, Autonomous Agents), then provide clear and concise answers to each question."

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
        user_input = input("You: Enter a topic (Agentic AI, AI, Autonomous Agents) for study assistance: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        # Call the function to get the questions and answers for the given topic
        response = run_study_assistant(user_input)
        
        # Display the assistant's response
        print("Assistant: " + response)


