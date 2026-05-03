def generate_quiz(context, llm):
    prompt = f"""
    Generate 3 multiple choice questions based on this:

    {context}

    Format:
    Question:
    A)
    B)
    C)
    Answer:
    """

    return llm.predict(prompt)