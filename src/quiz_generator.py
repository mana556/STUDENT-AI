import json
import re

def generate_quiz(context, llm, topic=None):
    MAX_CONTEXT_LENGTH = 4000  # Reduced to avoid 413 errors

    # Truncate context if too long
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH] + "... [truncated]"

    prompt = f"""Create 10 multiple choice questions on {topic if topic else "the provided text"} based on:

{context}

Return ONLY valid JSON array (no markdown, no extra text). Example format:
[{{"question":"What is X?","options":["A) option1","B) option2","C) option3"],"answer":"A"}}]"""

    return llm.predict(prompt)


def parse_quiz_output(quiz_text):
    """Parse quiz JSON with fallback to text-based parsing."""
    text = quiz_text.strip()
    
    # Remove markdown code blocks
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    
    # Try direct JSON parsing
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and all(isinstance(q, dict) for q in parsed):
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Fallback: extract JSON array from text
    try:
        json_match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            if isinstance(parsed, list) and all(isinstance(q, dict) for q in parsed):
                return parsed
    except (json.JSONDecodeError, AttributeError):
        pass
    
    return []
