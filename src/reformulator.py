"""
reformulator.py
Rewrites input claim using Groq LLM to include negation and null-result language.
This improves contradiction retrieval coverage for the queryreform strategy.
"""

from groq import Groq
from src.config import GROQ_API_KEY, GROQ_REFORMULATOR_MODEL


def reformulate_query(claim: str) -> str:
    """
    Takes a biomedical claim and returns a reformulated query
    that includes both confirming and disconfirming language.
    This helps dense retrieval surface contradicting abstracts.
    """
    import os
    from openai import OpenAI
    from groq import Groq
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
        )
    else:
        client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""You are a biomedical literature search expert.

Your task: rewrite the claim below into a search query that will retrieve BOTH supporting and contradicting scientific abstracts.

Rules:
- Include the original claim language
- Add negation variants: "no effect", "no significant effect", "failed to show", "did not improve", "null result"
- Add uncertainty language: "conflicting evidence", "inconsistent findings"
- Keep it under 60 words
- Return only the reformulated query, nothing else

Claim: {claim}

Reformulated query:"""

    response = client.chat.completions.create(
        model=GROQ_REFORMULATOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    test_claims = [
        "Vitamin D supplementation improves depression symptoms",
        "Omega-3 fatty acids reduce cardiovascular disease risk",
        "Intermittent fasting is more effective than caloric restriction for weight loss"
    ]

    for claim in test_claims:
        print(f"Original:     {claim}")
        reformulated = reformulate_query(claim)
        print(f"Reformulated: {reformulated}")
        print()
