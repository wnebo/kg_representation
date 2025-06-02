generate_entity_types = '''You are given a list of natural language questions (queries) that will be used to guide knowledge graph construction. Your task is to analyze these queries and summarize the types of entities and the types of relationships that will be needed to answer them.

Please follow these instructions:

1. Read through the entire query list.
2. Identify the kinds of entities that are likely to appear in the answers (e.g., "person", "organization", "product", "country", "date", etc.)
3. Identify the types of relationships implied by the questions (e.g., "founded_by", "located_in", "ceo_of", "produces", etc.)
4. Try to use general but informative type names. Avoid repeating the same type under different names unless necessary.
5. Output your response in the following JSON format:

```json
{
  "entity_types": ["..."],
  "relation_types": ["..."]
}
'''