"""Structure-aware deterministic chunking for cleaned documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.domain.models.chunk import Chunk
from app.domain.models.document import Document
from app.utils.hashing import sha256_text
from app.utils.ids import make_chunk_id

_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*([-*+]|(\d+[.)]))\s+.+")
_CODE_FENCE_RE = re.compile(r"^\s*```")


@dataclass(frozen=True)
class SemanticChunkingOptions:
    """Configurable options for deterministic semantic chunking."""

    strategy: str = "semantic"
    chunk_size_min: int = 300
    chunk_size_max: int = 1800
    chunk_overlap: int = 250
    max_overlap_units: int = 3
    min_chunk_chars: int = 1
    max_chunk_chars: int = 1800
    overlap_chars: int = 250
    use_semantic_boundaries: bool = False
    similarity_threshold: float = 0.55
    boundary_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    unit_embed_batch_size: int = 32


@dataclass(frozen=True)
class TextUnit:
    """Internal chunking unit built from a document block."""

    unit_index: int
    text: str
    unit_type: str
    start_char: int
    end_char: int
    heading_path: list[str]


@dataclass(frozen=True)
class _TextBlock:
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class _ChunkCandidate:
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    heading_path: list[str]
    unit_types: list[str]
    unit_count: int


class SemanticChunker:
    """Chunker that combines structure-aware splitting with deterministic packing."""

    CHUNKER_NAME = "SemanticChunker"

    def __init__(self, options: SemanticChunkingOptions) -> None:
        self._options = options
        self._boundary_model: Any | None = None

    def chunk(self, document: Document) -> list[Chunk]:
        """Split one cleaned document into retrieval-ready chunks."""
        if not document.text.strip():
            return []

        units = self._build_units(document.text)
        if not units:
            return []

        candidates = self._assemble_chunk_candidates(units)
        chunks: list[Chunk] = []
        seen_signatures: set[tuple[int, int, str]] = set()

        for candidate in candidates:
            text = candidate.text.strip()
            if not text:
                continue

            content_hash = sha256_text(text)
            signature = (candidate.char_start, candidate.char_end, content_hash)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            chunk_id = make_chunk_id(
                document_id=document.id,
                chunk_index=candidate.chunk_index,
                start_char=candidate.char_start,
                end_char=candidate.char_end,
                chunk_text_hash=content_hash,
            )
            token_count = self._estimate_token_count(text)
            metadata = self._build_chunk_metadata(document, candidate, token_count)
            section = candidate.heading_path[-1] if candidate.heading_path else None

            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    text=text,
                    chunk_index=candidate.chunk_index,
                    token_count=token_count,
                    content_hash=content_hash,
                    chunking_strategy=self._options.strategy,
                    heading_path=candidate.heading_path,
                    section=section,
                    char_start=candidate.char_start,
                    char_end=candidate.char_end,
                    metadata=metadata,
                )
            )

        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk all provided documents and flatten into a single list."""
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk(document))
        return chunks

    def _build_units(self, text: str) -> list[TextUnit]:
        blocks = self._split_blocks(text)
        units: list[TextUnit] = []
        heading_path: list[str] = []
        unit_index = 0

        for block in blocks:
            unit_type = self._classify_block(block.text)
            block_units: list[TextUnit] = []

            if unit_type == "heading":
                heading_level, heading_text = self._extract_heading(block.text)
                heading_path = self._updated_heading_path(heading_path, heading_level, heading_text)
                block_units = self._units_from_block(
                    block=block,
                    unit_type="heading",
                    heading_path=heading_path,
                    starting_unit_index=unit_index,
                    split_sentences=False,
                )
            elif unit_type == "list":
                block_units = self._units_from_block(
                    block=block,
                    unit_type="list",
                    heading_path=heading_path,
                    starting_unit_index=unit_index,
                    split_sentences=False,
                )
            elif unit_type == "code":
                block_units = self._units_from_block(
                    block=block,
                    unit_type="code",
                    heading_path=heading_path,
                    starting_unit_index=unit_index,
                    split_sentences=False,
                )
            else:
                split_sentences = len(block.text.strip()) > self._options.chunk_size_max
                block_units = self._units_from_block(
                    block=block,
                    unit_type="paragraph",
                    heading_path=heading_path,
                    starting_unit_index=unit_index,
                    split_sentences=split_sentences,
                )

            unit_index += len(block_units)
            units.extend(block_units)

        return units

    def _split_blocks(self, text: str) -> list[_TextBlock]:
        blocks: list[_TextBlock] = []
        lines = text.splitlines(keepends=True)
        block_start: int | None = None
        cursor = 0

        for line in lines:
            if line.strip():
                if block_start is None:
                    block_start = cursor
            else:
                if block_start is not None:
                    block_end = cursor
                    block_text = text[block_start:block_end]
                    if block_text.strip():
                        blocks.append(
                            _TextBlock(
                                text=block_text,
                                start_char=block_start,
                                end_char=block_end,
                            )
                        )
                    block_start = None
            cursor += len(line)

        if block_start is not None:
            block_text = text[block_start : len(text)]
            if block_text.strip():
                blocks.append(
                    _TextBlock(
                        text=block_text,
                        start_char=block_start,
                        end_char=len(text),
                    )
                )
        return blocks

    def _classify_block(self, block_text: str) -> str:
        stripped = block_text.strip()
        lines = [line for line in stripped.splitlines() if line.strip()]

        if not lines:
            return "paragraph"
        if _MARKDOWN_HEADING_RE.match(lines[0]):
            return "heading"
        if self._is_title_like_line(stripped):
            return "heading"
        if self._looks_like_code(lines):
            return "code"
        if self._looks_like_list(lines):
            return "list"
        return "paragraph"

    def _is_title_like_line(self, text: str) -> bool:
        line = text.strip()
        if "\n" in line or len(line) > 80:
            return False
        if re.search(r"[.!?:;]$", line):
            return False
        words = line.split()
        if len(words) > 10:
            return False
        alpha_words = [w for w in words if re.search(r"[A-Za-z]", w)]
        if not alpha_words:
            return False
        titled = sum(1 for w in alpha_words if w[:1].isupper())
        return titled / len(alpha_words) >= 0.7

    def _looks_like_list(self, lines: list[str]) -> bool:
        if len(lines) == 1:
            return bool(_BULLET_RE.match(lines[0]))
        list_lines = sum(1 for line in lines if _BULLET_RE.match(line))
        return list_lines >= max(1, int(len(lines) * 0.6))

    def _looks_like_code(self, lines: list[str]) -> bool:
        if any(_CODE_FENCE_RE.match(line) for line in lines):
            return True
        if any(line.startswith(("    ", "\t")) for line in lines):
            return True
        symbols = sum(1 for line in lines if re.search(r"[{}();=<>]", line))
        return symbols >= max(2, len(lines) // 2)

    def _extract_heading(self, block_text: str) -> tuple[int, str]:
        first_line = block_text.strip().splitlines()[0].strip()
        markdown_match = _MARKDOWN_HEADING_RE.match(first_line)
        if markdown_match:
            level = len(markdown_match.group(1))
            heading_text = markdown_match.group(2).strip()
            return level, heading_text
        return 1, first_line

    def _updated_heading_path(
        self,
        current_path: list[str],
        heading_level: int,
        heading_text: str,
    ) -> list[str]:
        trimmed = current_path[: max(0, heading_level - 1)]
        return [*trimmed, heading_text]

    def _units_from_block(
        self,
        *,
        block: _TextBlock,
        unit_type: str,
        heading_path: list[str],
        starting_unit_index: int,
        split_sentences: bool,
    ) -> list[TextUnit]:
        spans: list[tuple[int, int]]
        if split_sentences:
            spans = self._sentence_spans(block.text, block.start_char)
            if not spans:
                spans = [(block.start_char, block.end_char)]
        else:
            spans = [(block.start_char, block.end_char)]

        max_unit_chars = max(1, min(self._options.max_chunk_chars, self._options.chunk_size_max))
        refined_spans: list[tuple[int, int]] = []
        for start, end in spans:
            refined_spans.extend(
                self._hard_split_span_if_needed(
                    block_text=block.text,
                    block_start_char=block.start_char,
                    absolute_start=start,
                    absolute_end=end,
                    max_unit_chars=max_unit_chars,
                )
            )

        units: list[TextUnit] = []
        for offset, (start, end) in enumerate(refined_spans):
            relative_start = start - block.start_char
            relative_end = end - block.start_char
            span_text = block.text[relative_start:relative_end]

            left_trim = len(span_text) - len(span_text.lstrip())
            right_trim = len(span_text) - len(span_text.rstrip())
            normalized_start = start + left_trim
            normalized_end = end - right_trim
            normalized_text = span_text.strip()
            if not normalized_text:
                continue

            units.append(
                TextUnit(
                    unit_index=starting_unit_index + offset,
                    text=normalized_text,
                    unit_type=unit_type,
                    start_char=normalized_start,
                    end_char=normalized_end,
                    heading_path=list(heading_path),
                )
            )
        return units

    def _sentence_spans(self, block_text: str, block_start_char: int) -> list[tuple[int, int]]:
        sentence_boundaries = list(re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", block_text))
        if not sentence_boundaries:
            return []

        spans: list[tuple[int, int]] = []
        span_start = 0
        for boundary in sentence_boundaries:
            span_end = boundary.start()
            if span_end > span_start:
                spans.append((block_start_char + span_start, block_start_char + span_end))
            span_start = boundary.end()

        if span_start < len(block_text):
            spans.append((block_start_char + span_start, block_start_char + len(block_text)))
        return spans

    def _hard_split_span_if_needed(
        self,
        *,
        block_text: str,
        block_start_char: int,
        absolute_start: int,
        absolute_end: int,
        max_unit_chars: int,
    ) -> list[tuple[int, int]]:
        span_length = absolute_end - absolute_start
        if span_length <= max_unit_chars:
            return [(absolute_start, absolute_end)]

        split_spans: list[tuple[int, int]] = []
        local_start = absolute_start

        while local_start < absolute_end:
            hard_end = min(local_start + max_unit_chars, absolute_end)
            if hard_end == absolute_end:
                split_spans.append((local_start, hard_end))
                break

            window_relative_start = local_start - block_start_char
            window_relative_end = hard_end - block_start_char
            window_text = block_text[window_relative_start:window_relative_end]
            split_at = window_text.rfind(" ")
            if split_at <= 0:
                split_spans.append((local_start, hard_end))
                local_start = hard_end
            else:
                boundary = local_start + split_at
                split_spans.append((local_start, boundary))
                local_start = boundary + 1

        return split_spans

    def _assemble_chunk_candidates(self, units: list[TextUnit]) -> list[_ChunkCandidate]:
        max_chars = self._options.max_chunk_chars
        min_chars = self._options.min_chunk_chars
        boundary_breaks = self._semantic_boundary_breaks(units)

        candidates_units: list[list[TextUnit]] = []
        current_units: list[TextUnit] = []
        current_len = 0

        for idx, unit in enumerate(units):
            if idx in boundary_breaks and current_units and current_len >= min_chars:
                candidates_units.append(current_units)
                current_units = []
                current_len = 0

            separator_len = 2 if current_units else 0
            projected_len = current_len + separator_len + len(unit.text)

            if current_units and projected_len > max_chars and current_len >= min_chars:
                candidates_units.append(current_units)
                current_units = self._build_overlap_units(current_units)
                current_len = self._joined_length(current_units)

            separator_len = 2 if current_units else 0
            projected_len = current_len + separator_len + len(unit.text)
            if current_units and projected_len > max_chars and current_len == 0:
                current_units = [unit]
                current_len = len(unit.text)
            else:
                current_units.append(unit)
                current_len = projected_len

        if current_units:
            candidates_units.append(current_units)

        candidates_units = self._merge_tiny_tail(candidates_units)
        return self._build_candidates(candidates_units)

    def _build_overlap_units(self, units: list[TextUnit]) -> list[TextUnit]:
        overlap_units: list[TextUnit] = []
        overlap_chars = 0
        for unit in reversed(units):
            if len(overlap_units) >= self._options.max_overlap_units:
                break
            if overlap_chars >= self._options.overlap_chars:
                break
            overlap_units.append(unit)
            overlap_chars += len(unit.text)
        overlap_units.reverse()
        return overlap_units

    def _joined_length(self, units: list[TextUnit]) -> int:
        if not units:
            return 0
        return sum(len(unit.text) for unit in units) + (2 * (len(units) - 1))

    def _merge_tiny_tail(self, candidate_units: list[list[TextUnit]]) -> list[list[TextUnit]]:
        if len(candidate_units) < 2:
            return candidate_units

        last = candidate_units[-1]
        if self._joined_length(last) >= self._options.min_chunk_chars:
            return candidate_units

        previous = candidate_units[-2]
        merged = [*previous, *last]
        if self._joined_length(merged) <= self._options.max_chunk_chars:
            candidate_units[-2] = merged
            candidate_units.pop()
        return candidate_units

    def _build_candidates(self, candidates_units: list[list[TextUnit]]) -> list[_ChunkCandidate]:
        candidates: list[_ChunkCandidate] = []

        for chunk_index, units in enumerate(candidates_units):
            if not units:
                continue
            chunk_text = "\n\n".join(unit.text for unit in units).strip()
            if not chunk_text:
                continue

            char_start = min(unit.start_char for unit in units)
            char_end = max(unit.end_char for unit in units)
            heading_path = self._choose_heading_path(units)
            unit_types = self._ordered_unique([unit.unit_type for unit in units])

            candidates.append(
                _ChunkCandidate(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    char_start=char_start,
                    char_end=char_end,
                    heading_path=heading_path,
                    unit_types=unit_types,
                    unit_count=len(units),
                )
            )
        return candidates

    def _choose_heading_path(self, units: list[TextUnit]) -> list[str]:
        for unit in reversed(units):
            if unit.heading_path:
                return unit.heading_path
        return []

    def _ordered_unique(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    def _estimate_token_count(self, text: str) -> int:
        return max(1, len(text.split()))

    def _build_chunk_metadata(
        self,
        document: Document,
        candidate: _ChunkCandidate,
        token_count: int,
    ) -> dict[str, Any]:
        return {
            "title": document.title,
            "source_path": document.source_path,
            "source_name": document.source_name,
            "url": document.url,
            "source_type": document.source_type,
            "document_content_hash": document.content_hash,
            "processing_stage": document.processing_stage.value,
            "chunking_strategy": self._options.strategy,
            "chunker_name": self.CHUNKER_NAME,
            "chunk_size_min": self._options.chunk_size_min,
            "chunk_size_max": self._options.chunk_size_max,
            "chunk_overlap": self._options.chunk_overlap,
            "char_count": len(candidate.text),
            "token_count": token_count,
            "unit_count": candidate.unit_count,
            "start_char": candidate.char_start,
            "end_char": candidate.char_end,
            "heading_path": candidate.heading_path,
            "contains_heading": "heading" in candidate.unit_types,
            "unit_types": candidate.unit_types,
            "content_hash": document.content_hash,
        }

    def _semantic_boundary_breaks(self, units: list[TextUnit]) -> set[int]:
        """Return unit indexes where a hard boundary should be inserted."""
        if not self._options.use_semantic_boundaries or len(units) < 2:
            return set()

        model = self._get_boundary_model()
        if model is None:
            return set()

        texts = [unit.text for unit in units]
        vectors = model.encode(  # type: ignore[no-any-return]
            texts,
            batch_size=self._options.unit_embed_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        boundaries: set[int] = set()
        for idx in range(1, len(vectors)):
            similarity = float(vectors[idx - 1] @ vectors[idx])
            if similarity < self._options.similarity_threshold:
                boundaries.add(idx)
        return boundaries

    def _get_boundary_model(self) -> Any | None:
        if self._boundary_model is not None:
            return self._boundary_model

        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            return None

        self._boundary_model = SentenceTransformer(self._options.boundary_embedding_model_name)
        return self._boundary_model
