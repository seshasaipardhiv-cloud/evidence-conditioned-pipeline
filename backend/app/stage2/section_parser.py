"""
section_parser.py

Section-aware text segmentation for academic papers.

Given a block of text (full text or abstract), identifies:
  - Abstract
  - Introduction
  - Background
  - Methods / Materials and Methods
  - Dataset / Data
  - Experimental Setup
  - Results
  - Discussion
  - Limitations
  - Conclusion

Rules:
  - Section boundaries are identified by heading-like lines.
  - Introduction/Discussion text is NOT treated as empirical evidence.
  - Only Results sections are used for direct empirical claim extraction.
  - Unknown/unlabelled content is placed in 'unstructured'.
"""

import re
from typing import Dict, Optional

# Ordered list of known section names and their canonical keys
_SECTION_PATTERNS = [
    ("abstract", re.compile(r"^\s*abstract\s*$", re.IGNORECASE)),
    ("introduction", re.compile(r"^\s*(?:\d[\.\d]*\s+)?introduction\s*$", re.IGNORECASE)),
    ("background", re.compile(r"^\s*(?:\d[\.\d]*\s+)?background\s*$", re.IGNORECASE)),
    ("related_work", re.compile(r"^\s*(?:\d[\.\d]*\s+)?related\s+work\s*$", re.IGNORECASE)),
    ("methods", re.compile(
        r"^\s*(?:\d[\.\d]*\s+)?(?:materials?\s+and\s+)?methods?\s*$", re.IGNORECASE)),
    ("dataset", re.compile(
        r"^\s*(?:\d[\.\d]*\s+)?(?:data(?:set)?s?|study\s+population)\s*$", re.IGNORECASE)),
    ("experimental_setup", re.compile(
        r"^\s*(?:\d[\.\d]*\s+)?experimental\s+(?:setup|design|protocol)\s*$", re.IGNORECASE)),
    ("results", re.compile(r"^\s*(?:\d[\.\d]*\s+)?results?\s*$", re.IGNORECASE)),
    ("results_and_discussion", re.compile(
        r"^\s*(?:\d[\.\d]*\s+)?results?\s+and\s+discussion\s*$", re.IGNORECASE)),
    ("discussion", re.compile(r"^\s*(?:\d[\.\d]*\s+)?discussion\s*$", re.IGNORECASE)),
    ("limitations", re.compile(r"^\s*(?:\d[\.\d]*\s+)?limitations?\s*$", re.IGNORECASE)),
    ("conclusion", re.compile(
        r"^\s*(?:\d[\.\d]*\s+)?conclusions?\s*$", re.IGNORECASE)),
    ("ablation", re.compile(
        r"^\s*(?:\d[\.\d]*\s+)?ablation\s+(?:study|studies|experiment)\s*$", re.IGNORECASE)),
]

# Sections from which empirical results can be drawn
EMPIRICAL_SECTIONS = frozenset({"results", "results_and_discussion", "experimental_setup", "ablation"})

# Sections that are background/motivation only
BACKGROUND_SECTIONS = frozenset({"introduction", "background", "related_work"})


class SectionParser:

    def parse(self, text: str) -> Dict[str, str]:
        """
        Segment text into labelled sections.
        Returns dict mapping section_key → section_text.
        Unlabelled content accumulates in 'unstructured'.
        """
        if not text or not text.strip():
            return {}

        lines = text.splitlines()
        sections: Dict[str, list] = {"unstructured": []}
        current_section = "unstructured"

        for line in lines:
            matched = self._match_section(line)
            if matched:
                current_section = matched
                if matched not in sections:
                    sections[matched] = []
            else:
                sections.setdefault(current_section, []).append(line)

        return {k: "\n".join(v).strip() for k, v in sections.items() if any(l.strip() for l in v)}

    def _match_section(self, line: str) -> Optional[str]:
        stripped = line.strip()
        if not stripped or len(stripped) > 120:
            return None
        for key, pattern in _SECTION_PATTERNS:
            if pattern.match(stripped):
                return key
        return None

    def is_empirical_section(self, section_key: str) -> bool:
        """Returns True if results can legitimately come from this section."""
        return section_key in EMPIRICAL_SECTIONS

    def is_background_section(self, section_key: str) -> bool:
        return section_key in BACKGROUND_SECTIONS

    def get_results_text(self, sections: Dict[str, str]) -> str:
        """Returns the concatenated text from all empirical sections."""
        parts = []
        for key in ("results", "results_and_discussion", "experimental_setup"):
            if key in sections:
                parts.append(sections[key])
        return "\n".join(parts)

    def get_limitations_text(self, sections: Dict[str, str]) -> Optional[str]:
        return sections.get("limitations") or sections.get("conclusion")

    def get_ablation_text(self, sections: Dict[str, str]) -> Optional[str]:
        return sections.get("ablation")
