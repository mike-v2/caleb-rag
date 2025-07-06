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

IDENTIFY_SECTION_LINES_SYSTEM_PROMPT = """
You are an expert at analyzing video transcripts. Your task is to identify the line numbers corresponding to the start and end of the main monologue and the question-and-answer (Q&A) sections from the provided numbered transcript. Return the output in JSON format.
"""

def get_identify_section_lines_prompt(numbered_transcript: str, video_id: str) -> str:
    return f"""
<transcript_data>
<video_id>{video_id}</video_id>
<numbered_transcript>
{numbered_transcript}
</numbered_transcript>
</transcript_data>

Please analyze the numbered transcript above. The stream follows a typical structure:
1.  **Intro/Announcements**: A brief opening, often with personal anecdotes, organizational news, and calls to like/subscribe.
2.  **Monologue**: A long, continuous speech by the presenter on a specific topic. This is the main content.
3.  **Roll Call**: A section where names and locations are read out. This should be considered part of the Q&A transition.
4.  **Q&A Section**: A segment where the presenter answers questions from the audience, often signaled by phrases like "let's get to the questions", "Super Chats", or "Rumble Rants".

Your task is to identify the **start and end line numbers** for the 'monologue' and 'q_a' sections.

**Rules**:
- The 'monologue' begins AFTER any initial announcements or self-introduction and just BEFORE roll call starts. Just BEFORE names and locations are read out. Pay attention to when the speaker begins their main topic.
- The 'q_a' section begins when the floor is explicitly opened for questions from the audience. This is often clearly stated by the host.
- Carefully listen for the host describing the structure of the show. They often say something like, "First I'll give my remarks, then we'll do a roll call, then I'll answer your questions." Use this as a strong guide.
- Ignore the intro and roll call sections when defining the boundaries.
- If a section does not exist, its value should be null.
- The `start_line` is inclusive, and the `end_line` is inclusive.

Return a single JSON object with the keys "monologue" and "q_a". For example:
```json
{{
  "monologue": {{ "start_line": 10, "end_line": 450 }},
  "q_a": {{ "start_line": 452, "end_line": 670 }}
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
- **CRITICAL RULE**: You MUST return the full, original text, just split into chunks. DO NOT summarize, shorten, or change the text in any way. The concatenation of all chunks in the output MUST be exactly equal to the original input text.
- The provided text is a **{chunk_type.upper()}**.
- You must break the text at natural topic shifts or logical pauses.
- Each chunk should be internally consistent and make sense on its own.
- Never split a sentence in the middle.
- Preserve original line breaks and formatting within the chunks.

**Output Format Rules**:
- Your entire response **MUST** be a single, valid JSON object and nothing else.
- **DO NOT** include the `<text_to_split>` tags or any other text from the prompt in your output.
- **DO NOT** wrap the JSON in markdown backticks like ```json.
- Ensure all string values are correctly escaped (e.g., use `\\n` for newlines, `\\"` for quotes).

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