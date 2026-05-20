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

# Load environment variables
load_dotenv()

# Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=200,
    temperature=0.5
)

# Convert to chat model
model = ChatHuggingFace(llm=llm)

# Messages
messages = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]

# Invoke model
result = model.invoke(messages)

# Store AI response
messages.append(
    AIMessage(content=result.content)
)

# Print all messages
print(messages)