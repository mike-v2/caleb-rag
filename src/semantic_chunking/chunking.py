import logging
import re
from typing import List, Dict, Tuple

from .config import ENCODING, MAX_TOKENS, TARGET_TOKENS, MONOLOGUE_OVERLAP_SENTENCES
from .llm_client import call_llm, extract_json_from_llm_response
from . import prompts

def count_tokens(text: str) -> int:
    """Counts the number of tokens in a string using the pre-loaded tokenizer."""
    return len(ENCODING.encode(text))

def identify_sections(transcript: str, video_id: str) -> Dict[str, str]:
    """Uses an LLM to identify and separate the monologue and Q&A sections."""
    prompt = prompts.get_identify_sections_prompt(transcript, video_id)
    response = call_llm(prompt, prompts.IDENTIFY_SECTIONS_SYSTEM_PROMPT)
    
    if not response:
        logging.error(f"[{video_id}] Failed to get a response for section identification.")
        return {"monologue": "", "q_a": ""}

    extracted_json = extract_json_from_llm_response(response)
    if extracted_json:
        return {
            "monologue": extracted_json.get("monologue", ""),
            "q_a": extracted_json.get("q_a", ""),
        }

    logging.warning(f"[{video_id}] Could not extract JSON for sections. Falling back to simple heuristics.")
    lines = transcript.splitlines()
    monologue_end = int(len(lines) * 0.8)
    return {
        "monologue": "\n".join(lines[:monologue_end]),
        "q_a": "\n".join(lines[monologue_end:]),
    }

def _recursive_chunk_internal(text: str, video_id: str, chunk_type: str) -> List[Dict]:
    """Internal recursive chunking logic."""
    token_count = count_tokens(text)
    logging.info(f"[{video_id}] Recursive chunking ({chunk_type}) on {token_count} tokens.")

    if token_count <= MAX_TOKENS:
        return [{"text": text, "token_count": token_count}]

    num_splits = max(2, (token_count + TARGET_TOKENS - 1) // TARGET_TOKENS)
    prompt = prompts.get_recursive_chunk_prompt(text, num_splits, chunk_type)
    response = call_llm(prompt, prompts.RECURSIVE_CHUNK_SYSTEM_PROMPT)

    if response:
        extracted_json = extract_json_from_llm_response(response)
        chunks = extracted_json.get("chunks", []) if extracted_json else []
        if chunks and len(chunks) > 1:
            final_chunks = []
            for chunk_text in chunks:
                final_chunks.extend(_recursive_chunk_internal(chunk_text, video_id, chunk_type))
            return final_chunks

    logging.warning(f"[{video_id}] LLM failed to split text. Applying naive split.")
    lines = text.splitlines()
    midpoint = len(lines) // 2
    part1 = "\n".join(lines[:midpoint])
    part2 = "\n".join(lines[midpoint:])
    
    final_chunks = []
    final_chunks.extend(_recursive_chunk_internal(part1, video_id, chunk_type))
    final_chunks.extend(_recursive_chunk_internal(part2, video_id, chunk_type))
    return final_chunks

def process_qa_section(qa_text: str, video_id: str) -> List[Dict]:
    """Processes Q&A text by extracting pairs and chunking large ones."""
    logging.info(f"[{video_id}] Starting Q&A processing on {count_tokens(qa_text)} tokens.")
    prompt = prompts.get_qa_extraction_prompt(qa_text)
    response = call_llm(prompt, prompts.QA_EXTRACTION_SYSTEM_PROMPT)

    if not response:
        logging.error(f"[{video_id}] Failed to get a response for Q&A pair extraction.")
        return []

    extracted_json = extract_json_from_llm_response(response)
    qa_pairs = extracted_json.get("pairs", []) if extracted_json else []

    if not qa_pairs:
        logging.warning(f"[{video_id}] No Q&A pairs extracted. Treating entire section as one chunk.")
        return _recursive_chunk_internal(qa_text, video_id, "q_a")

    all_chunks = []
    for i, pair in enumerate(qa_pairs):
        q = pair.get('q', '').strip()
        a = pair.get('a', '').strip()
        if not q or not a:
            logging.warning(f"[{video_id}] Skipping malformed Q&A pair #{i+1}: {pair}")
            continue

        text = f"Question: {q}\nAnswer: {a}"
        if count_tokens(text) > MAX_TOKENS:
            chunks = _recursive_chunk_internal(text, video_id, "q_a")
            all_chunks.extend(chunks)
        else:
            all_chunks.append({"text": text, "token_count": count_tokens(text)})
    return all_chunks

def _find_chunk_start_time(chunk_text: str, source_text: str, time_map: List[Tuple[int, float]]) -> float:
    """Finds the start time of a chunk based on its text."""
    try:
        start_char_index = source_text.find(chunk_text)
        if start_char_index == -1:
            return 0.0
        
        # Find the latest time map entry before or at the chunk's start character
        # Assumes time_map is sorted by character index
        best_match = 0.0
        for char_index, time in time_map:
            if char_index <= start_char_index:
                best_match = time
            else:
                break
        return best_match
    except Exception as e:
        logging.warning(f"Could not determine start time for chunk: {e}")
        return 0.0

def process_video(video_data: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Full processing pipeline for a single video."""
    video_id = video_data.get('video_id', 'unknown_video')
    transcript_json = video_data.get('transcript_json', [])

    if not transcript_json:
        logging.warning(f"[{video_id}] `transcript_json` is empty or missing, skipping.")
        return [], []

    # 1. Concatenate transcript and create time map
    full_transcript_text = ""
    time_map = []
    for item in transcript_json:
        text_segment = item.get('text', '') + ' '
        start_time = item.get('start', 0.0)
        time_map.append((len(full_transcript_text), start_time))
        full_transcript_text += text_segment

    logging.info(f"--- Processing Video ID: {video_id} ---")
    sections = identify_sections(full_transcript_text, video_id)

    # 2. Process Monologue
    monologue_chunks = []
    monologue_text = sections.get("monologue")
    if monologue_text:
        raw_chunks = _recursive_chunk_internal(monologue_text, video_id, "monologue")
        for chunk in raw_chunks:
            start_time = _find_chunk_start_time(chunk['text'], full_transcript_text, time_map)
            chunk['start_time'] = start_time
            monologue_chunks.append(chunk)
        logging.info(f"[{video_id}] Created {len(monologue_chunks)} monologue chunks.")

    # 3. Process Q&A
    qa_chunks = []
    qa_text = sections.get("q_a")
    if qa_text:
        raw_chunks = process_qa_section(qa_text, video_id)
        for chunk in raw_chunks:
            start_time = _find_chunk_start_time(chunk['text'], full_transcript_text, time_map)
            chunk['start_time'] = start_time
            qa_chunks.append(chunk)
        logging.info(f"[{video_id}] Created {len(qa_chunks)} Q&A chunks.")

    return monologue_chunks, qa_chunks 