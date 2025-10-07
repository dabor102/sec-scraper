import re
from dateutil.parser import parse as parse_date
from dateutil.parser._parser import ParserError
from sec_parser.processing_steps.abstract_classes.processing_context import ElementProcessingContext
import logging as logger

def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace while preserving content."""
    if not text:
        return ""
    return ' '.join(text.split())


def is_header_row(text: str, context: ElementProcessingContext) -> bool:
    """
    Determines if a text string is a table header by checking for common
    financial date-related phrases or parseable dates.
    """
    # Clean and normalize the text for consistent matching
    normalized_text = ' '.join(text.strip().lower().split())

    # Rule 1: Check for common financial reporting period phrases
    period_pattern = re.compile(
        r"\b(three|six|nine|twelve)\s+(months|month)\s+ended\b"
    )
    fiscal_year_pattern = re.compile(r"\bfiscal\s+year\s+ended\b")
    if (
        period_pattern.search(normalized_text) or
        fiscal_year_pattern.search(normalized_text)
    ):
        return True

    # Rule 2: Use the dateutil library to robustly check for dates
    # We check substrings to see if any part of the text is a valid date
    # (e.g., to find "June 30, 2025" within "For the Period Ended June 30, 2025")
    words = normalized_text.split()
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            substring = " ".join(words[i:j])
            # Skip very short substrings that are unlikely to be dates
            if len(substring) < 5:
                continue
            try:
                # Attempt to parse the substring into a date
                parse_date(substring)
                # If it succeeds without error, we've found a date
                return True
            except (ParserError, ValueError, OverflowError, TypeError):
                # This substring is not a date, so we continue
                pass

    return False

