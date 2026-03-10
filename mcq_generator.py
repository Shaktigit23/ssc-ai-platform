from langchain_openai import ChatOpenAI

def generate_mcq(text):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
        )

    prompt = f"""
Generate 5 MCQ questions for class 6 students.

Text:
{text}

Format:
Question
A
B
C
D
Answer
"""

    response = llm.invoke(prompt)

    return response.content