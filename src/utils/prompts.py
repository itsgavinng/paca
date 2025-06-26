"""
Prompts for the Dual-Agent Perspective Discovery algorithm.

This module contains all prompt templates used in Stage 1 of the algorithm,
organized by the agent (DAgent/CAgent) and step in the process.
"""

# DAgent Prompts (Discovery Agent)

DAGENT_DISCOVER_DIMENSIONS = """
You are a Discovery Agent identifying context dimensions that influence HOW to answer questions, guided by causal validity principles.

Question: {question}

Discover dimensions that satisfy:
• **Temporal Precedence**: Exist BEFORE the question, independent of answer content (NOT the answer itself)
• **Factual Grounding**: Based on verifiable, evidence-based factors
• **Causal Relevance**: Influence the methodology/approach to answering, not the answer content

For each dimension consider: What contextual factor causally influences HOW we approach answering? How do different values lead to different methodologies? Does this exist independently of what the answer might be?

Cast a wide net across multiple domains while prioritizing strong empirical foundations and avoiding confounding factors.

[
  {{
    "name": "dimension_name",
    "description": "Brief description of the dimension",
    "justification": "Why this dimension is causally important for the answering approach",
    "score": 0.9
  }},
  ...
]

Rank by causal validity and importance (highest to lowest score).

{format_instructions}
"""

DAGENT_DISCOVER_PERSPECTIVES = """
You are a Discovery Agent identifying specific values within a context dimension, guided by causal validity principles.

Question: {question}
Dimension: {dimension_name} - {dimension_description}
Justification: {dimension_justification}

Discover perspectives (values) within this dimension that are:
• **Dimensional Consistency**: Comparable and measurable within the dimension
• **Temporal Precedence**: Independent of question outcome, DO NOT contain answer content
• **Factual Grounding**: Based on verifiable, evidence-based distinctions

For each perspective consider: What causally relevant value within this dimension (that is NOT the answer)? How does this systematically influence the APPROACH to answering? Is this a contextual condition rather than answer component? Can this be verified independently?

Seek genuine causal differences (not correlations), ensure mutual exclusivity where possible, prioritize empirical foundations, consider confounding factors and measurability.

[
  {{
    "value": "specific_perspective_value",
    "description": "Description with causal considerations",
    "justification": "Why this leads to causally different answering approaches"
  }},
  ...
]

Aim for 3-5 distinct, causally meaningful perspectives covering important variations.

{format_instructions}
"""

DAGENT_ASSIGN_WEIGHTS = """
You are a Discovery Agent assigning importance weights based on evidence quality and factual foundation.

Question: {question}
Dimension: {dimension_name} - {dimension_description}
Perspectives: {perspectives_json}

**PRIORITY CRITERIA (in order):**
1. **Factual Foundation**: Grounded in verifiable facts, documented evidence, established data
2. **Evidence Availability**: Empirical support, research, documented cases exist
3. **Verification Potential**: Can be objectively verified and validated
4. **Real-World Grounding**: Based on actual events, people, or phenomena rather than speculation
5. **Data-Driven Support**: Quantifiable and measurable with concrete evidence

**WEIGHTING PRINCIPLES:**
- Assign higher weights to perspectives with stronger empirical support
- Prioritize evidence-based distinctions over speculative or fictional ones
- Consider the quality and availability of supporting documentation
- Favor perspectives that lead to verifiable, accurate conclusions
- Ensure weight distribution reflects the strength of factual foundation

Weights must sum to 1.0 and be justified by evidence quality assessment.

{{
  "perspective_weights": {{
    "perspective_1_value": 0.4,
    "perspective_2_value": 0.35,
    "perspective_3_value": 0.25
  }},
  "justification": "Weight distribution based on evidence quality, factual foundation, and verification potential for each perspective"
}}

{format_instructions}
"""

# CAgent Prompts (Criteria Agent)

CAGENT_TEST_DIMENSIONS = """
You are a Criteria Agent CRITICALLY evaluating proposed dimensions against strict causal validity criteria.

Question: {question}
Proposed Dimensions: {dimensions_json}

**MANDATE: Be RIGOROUS and CRITICAL. Reject or heavily penalize dimensions that fail standards.**

**STRICT CRITERIA (all must pass):**
1. **Temporal Precedence**: Exists BEFORE question, about CONTEXT/METHODOLOGY not ANSWER CONTENT. REJECT dimensions containing answers or being the thing asked about.
2. **Factual Grounding**: Verifiable, objective, empirical. REJECT speculation or unverifiable assumptions.
3. **Causal Relevance**: Influences answering methodology, not answer content. REJECT dimensions that don't affect how we approach the question.

**CRITICAL ASSESSMENT:** Examine causal validity vs correlation, confounding risk, measurability, independence from outcome, empirical foundation.

**SCORING:** 0.9-1.0 (exceptional), 0.7-0.8 (good), 0.5-0.6 (moderate concerns), 0.1-0.4 (poor), 0.0 (invalid/reject).

Better to reject questionable dimensions than accept invalid ones.

[
  {{
    "name": "dimension_name", 
    "description": "description",
    "justification": "CRITICAL assessment: causal validity evaluation, criteria compliance/violations, score reasoning",
    "score": 0.8
  }},
  ...
]

{format_instructions}
"""

CAGENT_TEST_PERSPECTIVES = """
You are a Criteria Agent CRITICALLY evaluating proposed perspectives against strict causal validity criteria.

Question: {question}
Dimension: {dimension_name} - {dimension_description}
Proposed Perspectives: {perspectives_json}

**MANDATE: Be RIGOROUS and CRITICAL. Reject or heavily penalize perspectives that fail standards.**

**STRICT CRITERIA (all must pass):**
1. **Dimensional Consistency**: Same measurable scale within dimension, comparable and aggregatable. REJECT inconsistent scales.
2. **Temporal Precedence**: Exists BEFORE question context, about CONTEXT/CONDITIONS not ANSWER CONTENT. REJECT perspectives that ARE the answer, contain answer components, or are specific entities/names/facts being asked about.
3. **Factual Grounding**: Objective, verifiable, empirical distinctions. REJECT speculation or arbitrary labels.

**ADDITIONAL CHECKS:** Mutual exclusivity, causal relevance (not correlations), empirical support, confounding assessment, measurement validity.

**ASSESSMENT GUIDELINES:** Perspectives must pass ALL criteria. Better to reject questionable ones than accept invalid ones. Look for causal mechanisms, not statistical associations. Eliminate redundancy.

BE CRITICAL AND RIGOROUS.

[
  {{
    "value": "perspective_value",
    "description": "description with causal validation", 
    "justification": "CRITICAL assessment: causal validity evaluation, criteria compliance/violations, acceptance/rejection reasoning"
  }},
  ...
]

{format_instructions}
"""

CAGENT_ASSESS_WEIGHTS = """
You are a Criteria Agent rigorously evaluating weight assignments based on evidence quality and factual foundation.

Question: {question}
Dimension: {dimension_name} - {dimension_description}
Perspectives and Weights: {perspective_weights_json}
DAgent's Justification: {dagent_justification}

**EVALUATION CRITERIA:**
1. **Evidence Quality**: Are weights supported by verifiable facts, research, documented data?
2. **Factual Foundation**: Do weight assignments reflect objective, evidence-based distinctions?
3. **Verification Potential**: Are higher weights given to more verifiable perspectives?
4. **Empirical Support**: Is the distribution backed by concrete evidence rather than speculation?
5. **Documentation Quality**: Are perspectives with better documentation weighted appropriately?

**ADJUSTMENT PRINCIPLES:**
- Increase weights for perspectives with stronger empirical support
- Decrease weights for speculative or poorly documented perspectives
- Redistribute to reflect evidence quality and factual foundation
- Ensure final weights correspond to objective verification potential
- Prioritize perspectives that enable accurate, evidence-based conclusions

**CRITICAL ASSESSMENT:**
Evaluate whether the weight distribution appropriately reflects the strength of evidence, quality of documentation, and potential for verification across all perspectives.

{{
  "perspective_weights": {{
    "perspective_1_value": 0.45,
    "perspective_2_value": 0.30,
    "perspective_3_value": 0.25
  }},
  "justification": "Evidence-based evaluation: (1) assessment of documentation quality, (2) verification potential analysis, (3) empirical support evaluation, (4) factual foundation-based redistribution"
}}

Weights must sum to 1.0 and reflect evidence quality hierarchy.

{format_instructions}
"""

# Stage 2 Prompts (Perspective Resolution)

GENERATE_COT_PROMPT = """
When considering from the perspective of "{perspective_value}" within the dimension of "{dimension}", generate {k} distinct chains of thought (reasoning approaches) for answering the question below.

Question: {question}

Each chain of thought should explicitly reason from this perspective. Focus on the logical steps and methodology that this specific perspective would use, not the final answer.

Return your response in this JSON format:
{{"chains_of_thought": ["reasoning chain 1", "reasoning chain 2", ...]}}

Generate exactly {k} diverse reasoning chains from this perspective.
"""

ANSWER_FROM_COT_PROMPT = """
When considering from the perspective of "{perspective_value}", use the chain of thought below to answer the question.

Question: {question}

Chain of Thought: {cot}

Following this reasoning chain from this specific perspective, provide your answer. If the perspective leads to uncertainty or inability to determine an answer, use phrases like "no data", "cannot be determined", "insufficient evidence", or "unknowable from this perspective".

{{"answer": "your specific, concise answer here"}}

Be concise and direct in your answer.
"""

AGGREGATE_ANSWERS_PROMPT = """
Synthesize the following perspective-based answers into a single coherent response. Weight the perspectives based on their significance values.

Question: {question}

Perspectives and their answers:
{perspectives_summary}

Provide a balanced synthesis that acknowledges the most significant perspectives while noting any important variations or caveats.

Return your response in this JSON format:
{{"final_answer": "your synthesized answer"}}
"""

ABSTAIN_TYPE1_PROMPT = """
The analysis reveals contradictory information across different perspectives. Explain why a definitive answer cannot be provided.

Question: {question}
Contradiction Details: {contradiction_details}

Provide an explanation of why abstaining is appropriate due to conflicting perspectives.

Return your response in this JSON format:
{{"final_answer": "explanation of why abstaining due to contradiction"}}
"""

ABSTAIN_TYPE2_PROMPT = """
The analysis reveals insufficient information across perspectives to provide a confident answer.

Question: {question}
Insufficiency Details: {insufficiency_details}

Provide an explanation of why abstaining is appropriate due to knowledge gaps.

Return your response in this JSON format:
{{"final_answer": "explanation of why abstaining due to insufficient information"}}
"""

 