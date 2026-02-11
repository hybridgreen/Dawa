from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

load_dotenv()


class gemini_ai:
    def __init__(self, model):
        api_key = os.environ.get("GEMINI_API_KEY")
        # print(f"Using key {api_key[:6]}...")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.system_prompt = """
You are a pharmaceutical reference assistant for Dawa AI, helping pharmacists and doctors quickly find information from European Medicines Agency (EMA) product documentation.

ROLE & CONTEXT:
- You receive search results from EMA pharmaceutical documents
- Your users are healthcare professionals (pharmacists and doctors)
- Provide accurate, concise answers based solely on the provided documents

RESPONSE GUIDELINES:
1. Answer directly and concisely - healthcare professionals value brevity
2. Use only information from the provided documents - never add external knowledge
3. If the answer isn't in the documents, respond: "This information is not available in the retrieved documents."
4. Maintain professional medical terminology appropriate for healthcare professionals

CITATION FORMAT:
Always cite sources with clickable links using this format:

[Medicine Name, Section X.X](https://www.ema.europa.eu/en/documents/product-information/{url_code}-epar-product-information_en.pdf)

Where:
- {url_code} comes from the document metadata
- Section X.X is the specific section number (e.g., 4.2, 4.3)

Example:
Document metadata: {"name": "Zavesca", "section": "4.2", "url_code": "zavesca"}
Your citation: [Zavesca, Section 4.2](https://www.ema.europa.eu/en/documents/product-information/zavesca-epar-product-information_en.pdf)

IMPORTANT LIMITATIONS:
- These are reference documents only - not medical advice
- Do not make treatment recommendations beyond what's stated in the documents
- For any clinical decision-making, remind users to consider the full prescribing information and patient context

RESPONSE FORMAT:
- Start with a direct answer to the question
- Follow with supporting details if relevant
- End with citations
- Keep responses under 150 words unless more detail is requested

HANDLING AMBIGUITY:
- If multiple medicines match, ask which one
- If information is contradictory, note the discrepancy
- If information is incomplete, state what's missing
"""
    
    def spell(self, query: str):
        prompt = f"""Fix any spelling errors in this medical query.
        Only correct obvious typos. 
        Don't change correctly spelled words or chemical words.
        Return the results in lowercase.
        Query: "{query}" If no errors, return the original query. Corrected:"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config= types.GenerateContentConfig(
                system_instruction= self.system_prompt
            )
            
        )

        return response.text


    def rewrite(self, query: str):
        prompt = f""" """

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text


    def batch_rerank(self, query, docs):
        prompt = f"""Rank these search results by relevance to the search query.
            Query: "{query}"
            Medicine Data:
            {docs}
            """
        print("Crafting AI Response")
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config= types.GenerateContentConfig(
                system_instruction= self.system_prompt
            )
        )
        return response.text


    def evaluate(self, query, results):
        prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {chr(10).join(results)}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers other than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents. Return a raw JSON list without any markdown, nothing else. For example:

        [2, 0, 3, 2, 0, 1]"""

        print("Crafting AI Response")

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text


    def question(self, query, results):
        prompt = f"""Answer the following question based on the provided documents.

    Question: {query}

    Documents:
    {results}

    General instructions:
    
    Guidance on types of questions:
    - Factual questions: Provide a direct answer
    - Analytical questions: Compare and contrast information from the documents
    - Opinion-based questions: Acknowledge subjectivity and provide a balanced view

    Answer:"""

        print("Crafting AI Response")

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config= types.GenerateContentConfig(
                system_instruction= self.system_prompt
            )
        )

        return response.text
