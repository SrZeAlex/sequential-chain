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

# Chain 2: Write compelling introduction
intro_template = ChatPromptTemplate.from_messages([
    ("system", """You are a skilled technical writer known for engaging openings.
    Write introductions that hook readers immediately."""),
    ("human", """Based on this outline, write a compelling introduction (200-250 words):

    OUTLINE:
    {outline}

    Requirements:
    - Start with a surprising statistic or provocative question
    - Establish relevance to {audience}
    - Preview the main points without spoilers
    - End with a smooth transition to content
    - Use {tone} tone""")
])

intro_chain = intro_template | model | StrOutputParser()

# Test introduction
print("=== STEP 2: WRITING INTRODUCTION ===")
introduction = intro_chain.invoke({
    "outline": outline,
    "audience": ARTICLE_CONFIG["target_audience"],
    "tone": ARTICLE_CONFIG["tone"]
})

print(introduction)
print(f"\nWord count: {len(introduction.split())}")
print(f"\n{'='*60}\n")

# Chain 3: Develop main content sections
content_template = ChatPromptTemplate.from_messages([
    ("system", """You are a technical content writer with deep expertise in software development.
    Write detailed, accurate, and engaging content sections."""),
    ("human", """Write the main content based on this outline and introduction:

    OUTLINE:
    {outline}

    INTRODUCTION:
    {introduction}

    Requirements:
    - Cover all key points from outline
    - Approximately 700-800 words
    - Include specific examples and code snippets where relevant
    - Use subheadings for each major section
    - Maintain {tone} tone
    - Target audience: {audience}""")
])

content_chain = content_template | model | StrOutputParser()

# Test main content
print("=== STEP 3: CREATING MAIN CONTENT ===")
main_content = content_chain.invoke({
    "outline": outline,
    "introduction": introduction,
    "tone": ARTICLE_CONFIG["tone"],
    "audience": ARTICLE_CONFIG["target_audience"]
})

print(main_content)
print(f"\nWord count: {len(main_content.split())}")
print(f"\n{'='*60}\n")

