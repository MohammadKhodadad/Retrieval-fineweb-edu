from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start_token: int
    end_token: int


def _approx_tokens(text: str) -> list[str]:
    # Cheap approximation: whitespace tokens. Good enough for chunking.
    return text.split()


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _split_sentences(text: str) -> list[str]:
    # Best-effort sentence splitting (no heavy NLP deps).
    # - Splits on sentence-ending punctuation followed by whitespace.
    # - Also splits on blank-line paragraph breaks.
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p and p.strip()]
    return parts


def chunk_text(doc_id: str, text: str, chunk_tokens: int = 320, overlap: int = 40) -> list[Chunk]:
    if overlap >= chunk_tokens:
        raise ValueError("overlap must be < chunk_tokens")

    text = (text or "").strip()
    if not text:
        return []

    # Sentence-aware chunking:
    # - Pack whole sentences up to ~chunk_tokens (whitespace-token approximation).
    # - Preserve overlap (approx tokens) while keeping sentence boundaries.
    sentences = _split_sentences(text)
    if not sentences:
        return []

    sent_tok_lens = [_approx_tokens(s) for s in sentences]
    sent_lens = [len(t) for t in sent_tok_lens]

    # Precompute sentence token ranges (approx) for start/end offsets.
    sent_starts: list[int] = []
    sent_ends: list[int] = []
    cur = 0
    for n in sent_lens:
        sent_starts.append(cur)
        cur += n
        sent_ends.append(cur)

    chunks: list[Chunk] = []
    idx = 0
    s_idx = 0
    n_sent = len(sentences)

    def _find_sentence_index_for_token(token_pos: int) -> int:
        # Return the sentence index that contains token_pos (or the last sentence before it).
        # Linear scan is fine (n_sent per doc is modest).
        for i in range(n_sent):
            if sent_starts[i] <= token_pos < sent_ends[i]:
                return i
            if sent_starts[i] > token_pos:
                return max(0, i - 1)
        return n_sent - 1

    while s_idx < n_sent:
        # Grow chunk by whole sentences until we hit the target.
        e_idx = s_idx
        tok_count = 0
        while e_idx < n_sent:
            next_len = sent_lens[e_idx]
            if tok_count > 0 and (tok_count + next_len) > chunk_tokens:
                break
            tok_count += next_len
            e_idx += 1

        if e_idx == s_idx:
            # Single sentence longer than chunk_tokens: fall back to token-window chunking
            # within that sentence (rare, but otherwise we'd create huge chunks).
            toks = sent_tok_lens[s_idx]
            step = chunk_tokens - overlap
            i = 0
            while i < len(toks):
                j = min(i + chunk_tokens, len(toks))
                chunk_txt = " ".join(toks[i:j]).strip()
                if chunk_txt:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc_id}::c{idx}",
                            doc_id=doc_id,
                            text=chunk_txt,
                            start_token=sent_starts[s_idx] + i,
                            end_token=sent_starts[s_idx] + j,
                        )
                    )
                    idx += 1
                if j == len(toks):
                    break
                i += step
            s_idx += 1
            continue

        start_token = sent_starts[s_idx]
        end_token = sent_ends[e_idx - 1]
        chunk_txt = " ".join(sentences[s_idx:e_idx]).strip()
        if chunk_txt:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::c{idx}",
                    doc_id=doc_id,
                    text=chunk_txt,
                    start_token=start_token,
                    end_token=end_token,
                )
            )
            idx += 1

        if e_idx >= n_sent:
            break

        # Next chunk start: aim for overlap tokens, but snap to sentence boundary.
        target_start_token = max(0, end_token - overlap)
        s_idx = _find_sentence_index_for_token(target_start_token)
        # Ensure forward progress (avoid infinite loops when overlap is large vs sentence size).
        if sent_starts[s_idx] == start_token:
            s_idx = min(n_sent, s_idx + 1)

    return chunks

