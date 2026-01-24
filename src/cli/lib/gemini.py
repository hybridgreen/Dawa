from dotenv import load_dotenv
import os
from google import genai

load_dotenv()


class gemini_ai:
    def __init__(self, model):
        api_key = os.environ.get("GEMINI_API_KEY")
        # print(f"Using key {api_key[:6]}...")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def spell(self, query: str):
        prompt = f"""Fix any spelling errors in this medical query.
        Only correct obvious typos. 
        Don't change correctly spelled words or chemical words.
        Return the results in lowercase.
        Query: "{query}" If no errors, return the original query. Corrected:"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text

    def rewrite(self, query: str):
        prompt = f"""Rewrite this movie search query to be more specific and searchable.

                    Original: "{query}"

                    Consider:
                    - Common movie knowledge (famous actors, popular films)
                    - Genre conventions (horror = scary, animation = cartoon)
                    - Keep it concise (under 10 words)
                    - It should be a google style search query that's very specific
                    - Don't use boolean logic

                    Examples:

                    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                    Rewritten query:"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text

    def expand(self, query: str):
        prompt = f"""Expand this movie search query with related terms.

    Add synonyms and related concepts that might appear in movie descriptions.
    Keep expansions relevant and focused.
    This will be appended to the original query.

    Examples:

    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"

    Query: "{query}"
    """

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text

    def rerank(self, query, doc):
        prompt = f"""Rate how well this result matches the search query.

                Query: "{query}"
                Drug name: {doc.get("name", "")}
                Therapeutic Area: {doc.get("therapeutic_area", "")}
                EMA Section: {doc.get("section", "")}
                section content: {doc.get({"text"}, "")}
                

                Consider:
                    - Direct relevance to query
                    - User intent (what they're looking for)
                    - Content appropriateness

                Rate 0-10 (10 = perfect match).
                Give me ONLY the number in your response, no other text or explanation.

                Score:"""
        print("Crafting AI Response")
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return float(response.text)

    def batch_rerank(self, query, docs):
        prompt = f"""Rank these movies by relevance to the search query.
            Query: "{query}"
            Movies:
            {docs}
            Return ONLY the IDs in order of relevance (best match first). Return as a raw JSON list without markdown , nothing else. For example:
            [75, 12, 34, 2, 1]
            """
        print("Crafting AI Response")
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
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

    def augment(self, query, docs):
        prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.
            Query: {query}
            Documents:
            {docs}

            Provide a comprehensive answer that addresses the query:"""

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
    - Answer should be targeted to Dawa Users, Dawa is a medical drug assiatant.
    - Answer directly and concisely
    - Use only information from the documents
    - If the answer isn't in the documents, say "I don't have enough information"
    - Cite sources when possible, the document IDs are EMA product numbers in the format EMEA/H/C/XXXXXX

    Guidance on types of questions:
    - Factual questions: Provide a direct answer
    - Analytical questions: Compare and contrast information from the documents
    - Opinion-based questions: Acknowledge subjectivity and provide a balanced view

    Answer:"""

        print("Crafting AI Response")

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text
