"""
Prompts for the Dual-Agent Aspect Discovery algorithm.

This module contains all prompt templates used in Stage 1 of the algorithm,
organized by the agent (DAgent/CAgent) and step in the process.
"""

# DAgent Prompts (Discovery Agent)

DAGENT_DISCOVER_DIMENSIONS = """
You are a Discovery Agent that identifies context dimensions that influence HOW to answer the below question.

Question: {question}

Discover dimensions that satisfy: 
• **Temporal Precedence**: Exist BEFORE the question, independent of answer content (NOT the answer itself) 
• **Factual Grounding**: Based on verifiable, evidence-based factors, not non-factual factors

Consider: How can a dimension causally influence HOW we approach answering? How do different aspects within that dimension shape the path to the answer?

Then rank the dimensions by their importance to the question (highest to lowest score). 

Return your response in this JSON format:

{format_instructions}
"""

DAGENT_DISCOVER_PERSPECTIVES = """
You are a Discovery Agent that identifies specific aspects within a context dimension, guided by causal validity principles.

Question: {question}

Dimension: {dimension_name} - {dimension_description}

Justification: {dimension_justification}

Discover aspects within this dimension that satisfy:

• **Dimensional Consistency**: Comparable and measurable within the dimension.

• **Temporal Precedence**: Exists before and independent of question outcome, DO NOT contain answer content.

• **Factual Grounding**: Based on verifiable, evidence-based distinctions, not non-factual assumptions.

Seek genuine causal differences (not correlations), ensure mutual exclusivity where possible, prioritise empirical foundations, consider confounding factors and measurability.

Aim for up to {max_aspects} (not neccessary this many) distinct, causally meaningful aspects covering important variations. Return your response in this JSON format:

{format_instructions}
"""

DAGENT_ASSIGN_WEIGHTS = """
You are a Discovery Agent that assigns importance weights based on evidence quality and factual foundation.

Question: {question}

Dimension: {dimension_name} - {dimension_description}

Perspectives: {perspectives_json}

**WEIGHTING CRITERIA**:

• **Factual Foundation**: Grounded in verifiable facts, documented evidence, established data.

• **Evidence Availability**: Empirical support, research, documented cases exist.

• **Verification Potential**: Can be objectively verified and validated.

• **Real-World Grounding**: Based on actual events, people, or phenomena rather than speculation.

• **Data-Driven Support**: Quantifiable and measurable with concrete evidence.

Weights must sum to 1.0 and be justified by evidence quality assessment.

Return your response in this JSON format:

{format_instructions}
"""

# CAgent Prompts (Criteria Agent)

CAGENT_TEST_DIMENSIONS = """
You are a Critical Agent that CRITICALLY evaluates proposed dimensions against strict causal validity criteria.

Question: {question}

Proposed Dimensions: {dimensions_json}

**Strict causal validity criteria (all must pass)**:

• **Temporal Precedence**: Exists BEFORE question, about CONTEXT/METHODOLOGY not ANSWER CONTENT. REJECT dimensions containing answers or being the thing asked about.

• **Factual Grounding**: Verifiable, objective, empirical. REJECT speculation or unverifiable assumptions.

**MANDATE**: Be RIGOROUS and CRITICAL. Reject or heavily penalise dimensions that fail standards. Better to reject questionable dimensions than accept invalid ones.

Re-rank the remaining qualified dimensions based on alignment with the strict causal validity criteria. SCORING: 0.9-1.0 (exceptional alignment), 0.7-0.8 (good alignment), 0.5-0.6 (moderate concerns), 0.1-0.4 (poor), 0.0 (invalid/reject).

Return your response in this JSON format:

{format_instructions}
"""

CAGENT_TEST_PERSPECTIVES = """
You are a Critical Agent that CRITICALLY evaluates the proposed aspects against strict causal validity criteria.

Question: {question}

Dimension: {dimension_name} - {dimension_description}

Proposed Aspects: {aspects_json}

**Strict causal validity criteria (all must pass)**:

• **Dimensional Consistency**: Same measurable scale within dimension, comparable and aggregatable. REJECT inconsistent scales.

• **Temporal Precedence**: Exists BEFORE question context, about CONTEXT/CONDITIONS not ANSWER CONTENT. REJECT aspects that ARE a potential answer, contain answer components, or are specific entities/names/facts being asked about.

• **Factual Grounding**: Objective, verifiable, empirical distinctions. REJECT speculation or arbitrary labels.

**MANDATE**: Be RIGOROUS and CRITICAL. Reject or heavily penalise aspects that fail standards.  Better to reject questionable ones than accept invalid ones. Look for causal mechanisms, not statistical associations. Eliminate redundancy.

Return your response in this JSON format:

{format_instructions}
"""

CAGENT_ASSESS_WEIGHTS = """
You are a Critical Agent that rigorously evaluates weight assignments based on evidence quality and factual foundation.

Question: {question}

Dimension: {dimension_name} - {dimension_description}

Aspects and Weights: {aspects_weights_json}

DAgent's Justification: {dagent_justifications}

ADJUSTMENT PRINCIPLES:

• Increase weights for perspectives with stronger empirical support

• Decrease weights for speculative or poorly documented perspectives

• Redistribute to reflect evidence quality and factual foundation

• Ensure final weights correspond to objective verification potential

• Prioritise perspectives that enable accurate, evidence-based conclusions

Evaluate whether the weight distribution appropriately reflects the strength of evidence, quality of documentation, and potential for verification across all perspectives. Weights must sum to 1.0 and reflect evidence quality hierarchy. 

Return your response in this JSON format:

{format_instructions}
"""

# Stage 2 Prompts (Perspective Resolution)

GENERATE_COT_PROMPT = """
When considering the aspect of "{aspect_value}" within the dimension of "{dimension}", generate a chain of thought for answering the question below.

Question: {question}

The chain of thought should explicitly reason in this aspect. Focus on the logical steps and methodology that this specific aspect would use, not the final answer.

Return your response in this format:

{{"CoT": "Your chain of thought in this aspect"}}
"""

ANSWER_FROM_COT_PROMPT = """
When considering the aspect of "{aspect_value}", use the chain of thought below to answer the question.

Question: {question}

Chain of Thought: {CoT}

Following this reasoning chain in this specific aspect, provide your answer. If the aspect leads to uncertainty or inability to determine an answer, use phrases like "no data", "cannot be determined", "insufficient evidence", or "unknowable".

Return your response in this JSON format:

{{"answer": "Your specific, concise answer here. Don't include any explanation or reasoning."}}
"""

AGGREGATE_ANSWERS_PROMPT = """
Synthesise the following aspect-based answers into a single coherent response. Prioritise the aspects with higher significance values.

Question: {question}

Aspects, their significance, and their corresponding answers: {aspects_summary}

Provide a balanced synthesis that acknowledges the overarching answer across the most significant aspects while noting any minor variations or caveats.

Return your response in this format:

{{"final_answer": "Your synthesised answer"}}
"""

ABSTAIN_TYPE1_PROMPT = """
The analysis reveals contradictory information across different aspects. Explain why a definitive answer cannot be provided.

Question: {question}

Knowledge Conflict Details: {conflict_details}

Provide an explanation of why abstaining is appropriate due to conflicting information.

Return your response in this format:

{{"final_answer": "explanation of abstention rationale"}}
"""

ABSTAIN_TYPE2_PROMPT = """
The analysis reveals insufficient knowledge across aspects to provide a confident answer.

Question: {question}

Insufficiency Details: {insufficiency_details}

Provide an explanation of why abstaining is appropriate because you don't have enough knowledge to answer the question.

Return your response in this format:

{{"final_answer": "explanation of abstention rationale"}}
"""

 