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

# Chain 1: Generate article outline
outline_template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert content strategist for technical blogs.
    Create well-structured article outlines that flow logically."""),
    ("human", """Create a detailed outline for an article:

    Topic: {topic}
    Target Audience: {audience}
    Target Length: {word_count} words
    Tone: {tone}

    Outline Format:
    1. Hook (what grabs attention)
    2. Key Points (3-4 main sections with subpoints)
    3. Examples/Case Studies needed
    4. Conclusion focus

    Make it specific and actionable.""")
])

outline_chain = outline_template | model | StrOutputParser()

# Test outline generation
print("=== STEP 1: GENERATING OUTLINE ===")
outline = outline_chain.invoke({
    "topic": ARTICLE_CONFIG["topic"],
    "audience": ARTICLE_CONFIG["target_audience"],
    "word_count": ARTICLE_CONFIG["word_count"],
    "tone": ARTICLE_CONFIG["tone"]
})

print(outline)
print(f"\n{'='*60}\n")
