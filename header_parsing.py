import re
from dateutil.parser import parse as parse_date
from dateutil.parser._parser import ParserError
from sec_parser.processing_steps.abstract_classes.processing_context import ElementProcessingContext
import logging as logger

def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace and invisible characters."""
    if not text:
        return ""
    
    # Define common invisible characters to remove, including the zero-width space (\u200b)
    invisible_chars = [
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\ufeff',  # Byte order mark
    ]
    
    # Remove all defined invisible characters
    for char in invisible_chars:
        text = text.replace(char, '')
        
    # Standard whitespace normalization
    return ' '.join(text.split())

def is_header_row(text: str, context: ElementProcessingContext) -> bool:
    """
    Determines if a text string is a table header by checking for common
    financial date-related phrases or parseable dates.
    """
    # Clean and normalize the text for consistent matching
    normalized_text = normalize_text(text).lower()
    #print(normalized_text) 

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
    
    # Rule 1.5: Check for standalone four-digit years
    if re.search(r'\b(19|20)\d{2}\b', normalized_text):
        return True

    # Rule 2: Use the dateutil library to robustly check for dates
    # This will now work because the years are separated by spaces.
    words = normalized_text.split()
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            substring = " ".join(words[i:j])
            try:
                # Attempt to parse the substring into a date
                parse_date(substring)
                # If it succeeds, we've found a date
                return True
            except (ParserError, ValueError, OverflowError, TypeError):
                # This substring is not a date, so we continue
                pass

    return False