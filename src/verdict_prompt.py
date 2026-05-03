"""
verdict_prompt.py
Builds the structured prompt for the LLM verdict generator.
The prompt instructs the LLM to present both supporting and
contradicting evidence separately with citations.
"""


VERDICT_TEMPLATE = """You are a biomedical evidence synthesis expert.

Your task is to analyze the retrieved scientific abstracts and produce a structured verdict for the given claim.

CLAIM: {claim}

RETRIEVED ABSTRACTS:
{abstracts}

Instructions:
- Read all abstracts carefully
- Identify which abstracts SUPPORT the claim
- Identify which abstracts CONTRADICT or show no effect for the claim
- If an abstract is not relevant, do not cite it

Produce your verdict in exactly this format:

VERDICT: [Supported / Contradicted / Contested / Equipoise / Insufficient Evidence]

SUPPORTING EVIDENCE:
[For each supporting abstract: one sentence summary of what it found and why it supports the claim. Cite as (PMID: <id>)]

CONTRADICTING EVIDENCE:
[For each contradicting abstract: one sentence summary of what it found and why it contradicts the claim. Cite as (PMID: <id>)]

MODERATING FACTORS:
[Note any conditions, populations, or dosages that affect when the claim holds or does not hold]

CONFIDENCE: [Low / Low-to-Moderate / Moderate / Moderate-to-High / High]
[One sentence explaining the confidence level based on study quality and consistency]
"""


def build_verdict_prompt(claim: str, retrieved: list) -> str:
    """
    Builds the full prompt string from claim and retrieved abstracts.
    
    Args:
        claim: the input biomedical claim
        retrieved: list of abstract dicts from retriever/reranker
    
    Returns:
        formatted prompt string ready to send to the LLM
    """
    abstracts_text = ""
    for i, r in enumerate(retrieved):
        abstracts_text += f"[{i+1}] PMID: {r['id']}\n"
        abstracts_text += f"Title: {r['title']}\n"
        abstracts_text += f"Abstract: {r['text']}\n\n"

    return VERDICT_TEMPLATE.format(
        claim=claim,
        abstracts=abstracts_text.strip()
    )


if __name__ == "__main__":
    sample_retrieved = [
        {
            "id": "12345678",
            "title": "Vitamin D and depression: a meta-analysis",
            "text": "We found no significant effect of Vitamin D supplementation on depression scores in a randomized controlled trial of 500 adults."
        },
        {
            "id": "87654321",
            "title": "Vitamin D deficiency linked to depressive symptoms",
            "text": "In a cohort study of 1200 patients, low Vitamin D levels were significantly associated with higher rates of clinical depression."
        }
    ]

    prompt = build_verdict_prompt(
        claim="Vitamin D supplementation improves depression symptoms",
        retrieved=sample_retrieved
    )

    print(prompt)
