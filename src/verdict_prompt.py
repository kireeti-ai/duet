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

STEP 1: EVIDENCE FILTERING
Before writing your summary, review the abstracts. Identify which abstracts (by PMID) are DIRECTLY about the core subject of the claim (e.g., if the claim is about "Depression", the abstract must mention depression or clinical mood disorders).
- If an abstract is only about a mechanism (e.g., inflammation) or a different condition (e.g., arthritis), it is IRRELEVANT. 
- If ZERO abstracts are directly relevant, your only output must be: "No directly relevant evidence found in the retrieved abstracts."

STEP 2: SUMMARY
Only if you found relevant abstracts in Step 1, provide an objective summary of the evidence for and against the claim. 
- DO NOT use speculative language ("could", "may", "potential") to bridge gaps.
- DO NOT infer that a general process applies to a specific disease unless explicitly stated.

Claim: {claim}

Retrieved abstracts:
{abstracts}
"""


BIASED_PROMPT = """You are a biomedical evidence synthesis expert.

Claim: {claim}

Retrieved abstracts:
{abstracts}

Based on the retrieved abstracts, what evidence supports this claim?
Summarise the supporting findings and state how confident you are.
"""


STRUCTURED_PROMPT = """You are a biomedical evidence synthesis expert.

CRITICAL RULES (ANTI-HALLUCINATION):
1. RELEVANCE CHECK: If an abstract does not mention the EXACT core subject of the claim, it must be DISQUALIFIED.
2. NO SPECULATION: Do NOT "stretch" mechanism papers (e.g., cell studies) to support clinical claims.
3. If ZERO abstracts are directly about the claim's core condition, you MUST select "Insufficient Evidence" for the VERDICT and leave EVIDENCE sections empty.
4. Do not begin your response with any preamble about reviewing abstracts. Start directly with VERDICT.
5. Reference evidence using [1], [2], [3] inline, matching the order of retrieved abstracts.
6. Do NOT output any HTML or markdown formatting tags. Output plain text only.

Claim: {claim}

Retrieved abstracts:
{abstracts}

Produce your verdict in exactly this format:

VERDICT: [Supported / Contradicted / Mixed / Insufficient Evidence]

SUMMARY:
[2-3 sentence plain English summary with inline [1][2][3] citations]

SUPPORTING EVIDENCE:
[plain text, or "None found"]

CONTRADICTING EVIDENCE:
[plain text, or "None found"]

CONFIDENCE: [High / Medium / Low] — [one sentence reason]
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
