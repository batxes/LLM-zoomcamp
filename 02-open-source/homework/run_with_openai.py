import openai
import os
from openai import OpenAI

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)
# do this first in terminal_ export OPENAI_API_KEY=dummy_key 


# Set the API base URL to the local Docker container
openai.base_url="http://localhost:11434/v1"

# the code above does not work. I can connect to ollama with localhost:11434 but not with v1. 

try:
    models = client.models.list()
    for model in models:
        print(model)

except openai.OpenAIError as e:
    print(f"An error occurred: {e}")


try:
    # Create a request to the model
    #response = client.chat.completions.create(model="ollama-gemma2b",
    #response = client.chat.completions.create(model="gpt-3.5-turbo-1106",
    response = client.chat.completions.create(model="tts-1",
    messages=[
        {"role": "user", "content": "What's the formula for energy?"}
    ],
    temperature=0.0)

    print(response.choices[0].message.content.strip())

except openai.OpenAIError as e:
    print(f"An error occurred: {e}")


