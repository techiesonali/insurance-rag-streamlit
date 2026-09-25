import os
import time

from langchain_google_genai import ChatGoogleGenerativeAI


api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    google_api_key=api_key,
    max_retries=3,
)


prompt = """
Explain Retrieval Augmented Generation in one short paragraph.
Focus on the relationship between retrieval, context, and generation.
"""

for attempt in range(1, 4):

    try:

        print(f"\nAttempt {attempt}/3...")

        response = llm.invoke(prompt)

        print("\n========== GEMINI RESPONSE ==========\n")
        print(response.content)

        break

    except Exception as e:

        print(f"\nRequest failed: {e}")

        if attempt < 3:
            print("Waiting before retrying...")
            time.sleep(5)
        else:
            print("\nGemini is currently unavailable. Please try again later.")