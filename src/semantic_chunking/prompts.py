IDENTIFY_SECTIONS_SYSTEM_PROMPT = """
You are an expert at analyzing video transcripts. Your task is to identify and extract the main monologue and the question-and-answer (Q&A) sections from the provided transcript. Return the output in JSON format.
"""

def get_identify_sections_prompt(transcript: str, video_id: str) -> str:
    return f"""
<transcript_data>
<video_id>{video_id}</video_id>
<transcript>
{transcript}
</transcript>
</transcript_data>

Please analyze the transcript above. Identify the primary monologue and the Q&A section.

The stream follows a typical structure:
1.  **Intro/Announcements**: A brief opening, which should be ignored.
2.  **Monologue**: A long, continuous speech by the presenter. This is the main content.
3.  **Roll Call**: A section where names and locations are read out, which should be ignored.
4.  **Q&A Section**: A segment where the presenter answers questions from the audience.

Extract the verbatim text for the 'monologue' and 'q_a' sections. If a section is not present, the value should be an empty string.

Return a single JSON object with the keys "monologue" and "q_a". For example:
```json
{{
  "monologue": "The full text of the monologue...",
  "q_a": "The full text of the Q&A section..."
}}
```
"""

RECURSIVE_CHUNK_SYSTEM_PROMPT = """
You are a text processing expert. Your task is to split a large block of text into smaller, coherent chunks based on the provided rules. Output the result as a JSON array of strings.
"""

def get_recursive_chunk_prompt(text: str, num_splits: int, chunk_type: str) -> str:
    return f"""
<text_to_split>
{text}
</text_to_split>

You must split the provided text into {num_splits} semantically coherent chunks.

**Instructions**:
- The provided text is a **{chunk_type.upper()}**.
- You must break the text at natural topic shifts or logical pauses.
- Each chunk should be internally consistent and make sense on its own.
- Never split a sentence in the middle.
- Preserve original line breaks and formatting within the chunks.

Return a single JSON object with a "chunks" key, containing a list of the text chunks. For example:
```json
{{
  "chunks": ["First part of the text...", "Second part...", "Third part..."]
}}
```
"""

QA_EXTRACTION_SYSTEM_PROMPT = """
You are an expert in parsing interview and Q&A transcripts. Your task is to identify and extract distinct question-answer pairs from the provided text. Return the result as a JSON object.
"""

def get_qa_extraction_prompt(qa_text: str) -> str:
    return f"""
<qa_text>
{qa_text}
</qa_text>

Please analyze the Q&A text above and extract all distinct question-and-answer pairs.

**Rules**:
- Each object in the output array should represent one full Q&A exchange.
- The 'q' key should contain the question, and the 'a' key should contain the corresponding answer.
- Include all related follow-up discussion with its original question.
- Preserve the exact wording from the transcript.

Return a single JSON object with a "pairs" key. The value should be a list of Q&A objects.
Example:
```json
{{
  "pairs": [
    {{
      "q": "First question text...",
      "a": "Answer to the first question."
    }},
    {{
      "q": "Second question...",
      "a": "Response to the second question, including any follow-up."
    }}
  ]
}}
```
""" 