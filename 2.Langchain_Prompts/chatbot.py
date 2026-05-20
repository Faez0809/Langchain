from dotenv import load_dotenv

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace
)

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# Load .env variables
load_dotenv()

# Hugging Face LLM
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=200,
    temperature=0.5
)

# Convert into chat model
model = ChatHuggingFace(llm=llm)

# Chat history
chat_history = [
    SystemMessage(content="You are a helpful AI assistant")
]

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # Add user message
    chat_history.append(
        HumanMessage(content=user_input)
    )

    # Generate response
    result = model.invoke(chat_history)

    # Save AI response
    chat_history.append(
        AIMessage(content=result.content)
    )

    print("AI:", result.content)

# Print full conversation
print(chat_history)