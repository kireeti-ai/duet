"""
verdict_prompt.py
Three prompt variants for INV-03 prompt sensitivity investigation.

NEUTRAL:    asks LLM to summarise evidence without structure
BIASED:     asks only for supporting evidence -- deliberately one-sided
STRUCTURED: asks for explicit sections with citations -- most grounded

Switch variant by changing PROMPT_VARIANT in config.py.
"""

from src.config import PROMPT_VARIANT


NEUTRAL_PROMPT = """You are a biomedical evidence synthesis expert.

Claim: {claim}

Retrieved abstracts:
{abstracts}

Summarise the evidence for and against this claim based on the retrieved abstracts.
Include a confidence assessment at the end.
"""


BIASED_PROMPT = """You are a biomedical evidence synthesis expert.

Claim: {claim}

Retrieved abstracts:
{abstracts}

Based on the retrieved abstracts, what evidence supports this claim?
Summarise the supporting findings and state how confident you are.
"""


STRUCTURED_PROMPT = """You are a biomedical evidence synthesis expert.

Claim: {claim}

Retrieved abstracts:
{abstracts}

Produce your verdict in exactly this format:

VERDICT: [Supported / Contradicted / Contested / Equipoise / Insufficient Evidence]

SUPPORTING EVIDENCE:
[For each supporting abstract: one sentence summary. Cite as (PMID: <id>)]

CONTRADICTING EVIDENCE:
[For each contradicting abstract: one sentence summary. Cite as (PMID: <id>)]

MODERATING FACTORS:
[Conditions, populations, or dosages that affect when the claim holds]

CONFIDENCE: [Low / Low-to-Moderate / Moderate / Moderate-to-High / High]
[One sentence explaining the confidence level]
"""


PROMPT_VARIANTS = {
    "neutral": NEUTRAL_PROMPT,
    "biased": BIASED_PROMPT,
    "structured": STRUCTURED_PROMPT
}


def build_verdict_prompt(claim: str, retrieved: list, variant: str = PROMPT_VARIANT) -> str:
    """
    Builds the full prompt from claim and retrieved abstracts.

    Args:
        claim: the input biomedical claim
        retrieved: list of abstract dicts from retriever/reranker
        variant: which prompt variant to use (neutral/biased/structured)

    Returns:
        formatted prompt string ready to send to the LLM
    """
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant: {variant}. Choose: neutral | biased | structured")

    abstracts_text = ""
    for i, r in enumerate(retrieved):
        abstracts_text += f"[{i+1}] PMID: {r['id']}\n"
        abstracts_text += f"Title: {r['title']}\n"
        abstracts_text += f"Abstract: {r['text']}\n\n"

    template = PROMPT_VARIANTS[variant]
    return template.format(
        claim=claim,
        abstracts=abstracts_text.strip()
    )


if __name__ == "__main__":
    sample_retrieved = [
        {
            "id": "12345678",
            "title": "Omega-3 and depression meta-analysis",
            "text": "We found no significant effect of omega-3 on depression scores in 500 adults."
        },
        {
            "id": "87654321",
            "title": "Omega-3 deficiency linked to depressive symptoms",
            "text": "Low omega-3 levels were significantly associated with higher rates of clinical depression."
        }
    ]

    claim = "Omega-3 supplementation reduces symptoms of depression"

    for variant in ["neutral", "biased", "structured"]:
        print(f"\n{'='*50}")
        print(f"VARIANT: {variant.upper()}")
        print('='*50)
        prompt = build_verdict_prompt(claim, sample_retrieved, variant=variant)
        print(prompt)
