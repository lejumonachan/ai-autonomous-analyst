import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv()

# Read API key
api_key = os.getenv("OPENAI_API_KEY")

# Check API key
if api_key is None or api_key.strip() == "":
    raise ValueError(
        "OPENAI_API_KEY not found inside .env file"
    )

# Create OpenAI client
client = OpenAI(api_key=api_key)


def ask_llm(prompt):

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",

            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a senior AI Data Analyst.

                    Analyze datasets professionally.

                    Give:
                    - Business insights
                    - Data quality analysis
                    - Trends
                    - Correlation observations
                    - ML recommendations
                    - Risk analysis
                    - Statistical observations
                    """
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"LLM Error: {e}"