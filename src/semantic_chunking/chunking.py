import logging
import re
import os
import time
from typing import List, Dict, Tuple, Optional

from .config import ENCODING, MAX_TOKENS, TARGET_TOKENS, MONOLOGUE_OVERLAP_SENTENCES, LOGS_DIR
from .llm_client import call_llm, extract_json_from_llm_response
from . import prompts

def count_tokens(text: str) -> int:
    """Counts the number of tokens in a string using the pre-loaded tokenizer."""
    return len(ENCODING.encode(text))

def identify_section_lines(transcript_json: List[Dict], video_id: str) -> Dict[str, Optional[Dict]]:
    """Uses an LLM to identify the line numbers for monologue and Q&A sections."""
    if not transcript_json:
        return {"monologue": None, "q_a": None}

    numbered_transcript = "\n".join(f"{i+1}: {item['text']}" for i, item in enumerate(transcript_json))
    
    prompt = prompts.get_identify_section_lines_prompt(numbered_transcript, video_id)
    response = call_llm(prompt, prompts.IDENTIFY_SECTION_LINES_SYSTEM_PROMPT)
    
    if response:
        extracted_json = extract_json_from_llm_response(response)
        if extracted_json:
            # Basic validation
            if "monologue" in extracted_json and "q_a" in extracted_json:
                logging.info(f"[{video_id}] Successfully identified section lines via LLM.")
                return extracted_json

    logging.warning(f"[{video_id}] Could not extract valid section lines via LLM. Falling back to 80/20 split.")
    total_lines = len(transcript_json)
    monologue_end = int(total_lines * 0.8)
    return {
        "monologue": {"start_line": 1, "end_line": monologue_end},
        "q_a": {"start_line": monologue_end + 1, "end_line": total_lines},
    }

def _recursive_chunk_internal(text: str, video_id: str, chunk_type: str) -> List[Dict]:
    """Internal recursive chunking logic."""
    token_count = count_tokens(text)
    logging.info(f"[{video_id}] Recursive chunking ({chunk_type}) on {token_count} tokens.")

    if not text.strip():
        return []

    if token_count <= MAX_TOKENS:
        return [{"text": text, "token_count": token_count}]

    # If text is too large for the LLM to return in one go, bypass LLM and split naively.
    # This avoids failed LLM calls and excessive sleeps.
    if token_count > 4000:  # Using a safe margin below the 4096 token output limit.
        logging.info(f"[{video_id}] Text too large for LLM-based split ({token_count} tokens). Applying naive split.")
        
        midpoint = len(text) // 2
        split_pos = text.rfind(' ', 0, midpoint)
        if split_pos == -1:
            split_pos = text.find(' ', midpoint)
        if split_pos == -1:
            split_pos = midpoint
        
        part1 = text[:split_pos]
        part2 = text[split_pos:]

        # Prevent infinite recursion if splitting fails to make progress
        if not part1.strip() or not part2.strip():
            logging.warning(f"[{video_id}] Naive split failed to make progress on {token_count} tokens. Returning as single chunk.")
            return [{"text": text, "token_count": token_count}]
        
        final_chunks = []
        final_chunks.extend(_recursive_chunk_internal(part1, video_id, chunk_type))
        final_chunks.extend(_recursive_chunk_internal(part2, video_id, chunk_type))
        return final_chunks

    # For text small enough to be processable by the LLM.
    num_splits = max(2, (token_count + TARGET_TOKENS - 1) // TARGET_TOKENS)
    prompt = prompts.get_recursive_chunk_prompt(text, num_splits, chunk_type)
    
    # Proactively sleep to avoid hitting rate limits, especially on token-heavy calls.
    time.sleep(2)
    response = call_llm(prompt, prompts.RECURSIVE_CHUNK_SYSTEM_PROMPT)

    if response:
        extracted_json = extract_json_from_llm_response(response)
        chunks = extracted_json.get("chunks", []) if extracted_json else []
        if chunks and len(chunks) > 1:
            final_chunks = []
            for chunk_text in chunks:
                # The recursive call here will likely hit the base case and not sleep again.
                final_chunks.extend(_recursive_chunk_internal(chunk_text, video_id, chunk_type))
            return final_chunks

    # Fallback for when the LLM fails on a processable-sized chunk.
    logging.warning(f"[{video_id}] LLM failed to split processable text. Applying naive split.")
    
    midpoint = len(text) // 2
    split_pos = text.rfind(' ', 0, midpoint)
    if split_pos == -1:
        split_pos = text.find(' ', midpoint)
    if split_pos == -1:
        split_pos = midpoint

    part1 = text[:split_pos]
    part2 = text[split_pos:]
    
    # Prevent infinite recursion if splitting fails to make progress
    if not part1.strip() or not part2.strip():
        logging.warning(f"[{video_id}] Naive split failed to make progress on {token_count} tokens. Returning as single chunk.")
        return [{"text": text, "token_count": token_count}]

    final_chunks = []
    final_chunks.extend(_recursive_chunk_internal(part1, video_id, chunk_type))
    final_chunks.extend(_recursive_chunk_internal(part2, video_id, chunk_type))
    return final_chunks

def process_qa_section(qa_text: str, video_id: str, qa_time_map: List[Tuple[int, float]]) -> List[Dict]:
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
        # Chunk the whole text and map times
        raw_chunks = _recursive_chunk_internal(qa_text, video_id, "q_a")
        final_chunks = []
        for chunk in raw_chunks:
            chunk['start_time'] = _map_chunk_to_start_time(chunk['text'], qa_text, qa_time_map)
            final_chunks.append(chunk)
        return final_chunks

    all_chunks = []
    for i, pair in enumerate(qa_pairs):
        q = pair.get('q', '').strip()
        a = pair.get('a', '').strip()
        if not q or not a:
            logging.warning(f"[{video_id}] Skipping malformed Q&A pair #{i+1}: {pair}")
            continue
        
        pair_text = f"Question: {q}\nAnswer: {a}"
        pair_start_time = _map_chunk_to_start_time(q, qa_text, qa_time_map)

        if count_tokens(pair_text) > MAX_TOKENS:
            # If the pair is too large, chunk it, but ensure start times are mapped correctly
            chunks = _recursive_chunk_internal(pair_text, video_id, "q_a")
            for chunk in chunks:
                # All sub-chunks of a pair get the start time of the question
                chunk['start_time'] = pair_start_time
                all_chunks.append(chunk)
        else:
            all_chunks.append({
                "text": pair_text,
                "token_count": count_tokens(pair_text),
                "start_time": pair_start_time
            })
    return all_chunks

def _map_chunk_to_start_time(chunk_text: str, source_text: str, time_map: List[Tuple[int, float]]) -> float:
    """Finds the start time of a chunk based on its text within a source text."""
    try:
        # Sanitize chunk text by taking the first 20 words to avoid hallucinated text
        search_text = " ".join(chunk_text.split()[:20])
        start_char_index = source_text.find(search_text)
        
        if start_char_index == -1:
            logging.warning(f"Could not find start of chunk in source text. Chunk: '{search_text[:100]}...'")
            return time_map[0][1] if time_map else 0.0
        
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

    logging.info(f"--- Processing Video ID: {video_id} ---")
    logging.info(f"[{video_id}] Transcript has {len(transcript_json)} lines. Identifying section lines...")
    
    section_lines = identify_section_lines(transcript_json, video_id)
    monologue_section = section_lines.get("monologue")
    qa_section = section_lines.get("q_a")

    # 1. Process Monologue
    monologue_chunks = []
    if monologue_section and monologue_section.get("start_line") and monologue_section.get("end_line"):
        logging.info(f"[{video_id}] Monologue identified: lines {monologue_section['start_line']}-{monologue_section['end_line']}.")
        m_start = monologue_section["start_line"] - 1
        m_end = monologue_section["end_line"]
        monologue_items = transcript_json[m_start:m_end]

        monologue_text = " ".join(item['text'] for item in monologue_items)
        
        # Log the entire monologue section to a file
        monologue_log_path = os.path.join(LOGS_DIR, f"{video_id}_monologue.txt")
        with open(monologue_log_path, 'w', encoding='utf-8') as f:
            f.write(f"--- MONOLOGUE SECTION (Lines {monologue_section['start_line']}-{monologue_section['end_line']}) ---\n")
            f.write(monologue_text)

        monologue_time_map = []
        offset = 0
        for item in monologue_items:
            monologue_time_map.append((offset, item.get('start', 0.0)))
            offset += len(item['text']) + 1

        logging.info(f"[{video_id}] Chunking monologue...")
        raw_chunks = _recursive_chunk_internal(monologue_text, video_id, "monologue")
        for chunk in raw_chunks:
            chunk['start_time'] = _map_chunk_to_start_time(chunk['text'], monologue_text, monologue_time_map)
            monologue_chunks.append(chunk)

        # Append the chunks to the log file
        with open(monologue_log_path, 'a', encoding='utf-8') as f:
            f.write("\n\n--- MONOLOGUE CHUNKS ---\n")
            for i, chunk in enumerate(monologue_chunks):
                f.write(f"\n--- Chunk {i+1} (Start: {chunk['start_time']:.2f}s, Tokens: {chunk['token_count']}) ---\n")
                f.write(chunk['text'])

        logging.info(f"[{video_id}] Created {len(monologue_chunks)} monologue chunks.")

    # 2. Process Q&A
    qa_chunks = []
    if qa_section and qa_section.get("start_line") and qa_section.get("end_line"):
        logging.info(f"[{video_id}] Q&A identified: lines {qa_section['start_line']}-{qa_section['end_line']}.")
        q_start = qa_section["start_line"] - 1
        q_end = qa_section["end_line"]
        qa_items = transcript_json[q_start:q_end]

        qa_text = " ".join(item['text'] for item in qa_items)
        
        # Log the entire Q&A section to a file
        qa_log_path = os.path.join(LOGS_DIR, f"{video_id}_q_a.txt")
        with open(qa_log_path, 'w', encoding='utf-8') as f:
            f.write(f"--- Q&A SECTION (Lines {qa_section['start_line']}-{qa_section['end_line']}) ---\n")
            f.write(qa_text)

        qa_time_map = []
        offset = 0
        for item in qa_items:
            qa_time_map.append((offset, item.get('start', 0.0)))
            offset += len(item['text']) + 1

        logging.info(f"[{video_id}] Chunking Q&A section...")
        qa_chunks = process_qa_section(qa_text, video_id, qa_time_map)
        
        # Append the chunks to the log file
        with open(qa_log_path, 'a', encoding='utf-8') as f:
            f.write("\n\n--- Q&A CHUNKS ---\n")
            for i, chunk in enumerate(qa_chunks):
                f.write(f"\n--- Chunk {i+1} (Start: {chunk['start_time']:.2f}s, Tokens: {chunk['token_count']}) ---\n")
                f.write(chunk['text'])
        
        logging.info(f"[{video_id}] Created {len(qa_chunks)} Q&A chunks.")

    return monologue_chunks, qa_chunks 