from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import os

# Initialize model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.7  # Balanced creativity
)

# Article configuration
ARTICLE_CONFIG = {
    "topic": "The Rise of AI Agents in Software Development",
    "target_audience": "software developers and engineering managers",
    "word_count": 1200,
    "tone": "informative yet conversational"
}
