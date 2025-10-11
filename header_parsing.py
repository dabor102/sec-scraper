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
    financial date-related phrases, units, or parseable dates.
    """
    normalized_text = normalize_text(text).lower()
    logger.info(normalized_text)

    if len(normalized_text)>250:
        logger.info("Long text -> NOT TITLE")
        return True

    # Rule 1: Check for common financial reporting period phrases
    asof_pattern = re.compile(r"\b(as of)\b")
    period_pattern = re.compile(
    r"\b((first|second|third|fourth)\s+)?(three|six|nine|twelve)\s+(months|month)(\s+ended)?\b"
    r"|\b((first|second|third|fourth)\s+)?quarter\b",
    re.IGNORECASE  # Use IGNORECASE for case-insensitivity
)
    fiscal_year_pattern = re.compile(r"\bfiscal\s+year\s+ended\b")
    
    if (
        asof_pattern.search(normalized_text) or
        period_pattern.search(normalized_text) or
        fiscal_year_pattern.search(normalized_text)
    ):
        logger.info("period match-> NO TITLE")
        return True
    
    # Rule 1.5: Check for unit declarations or qualifiers
    units_pattern = re.compile(
    # This now checks for EITHER a literal dollar sign
    # OR one of the specified whole words.
    r"(\$|\b(thousands|millions|billions|unaudited|per share)\b)"
)
    if units_pattern.search(normalized_text):
        logger.info("units match-> NO TITLE")
        return True 

    
    # Rule 2: Check for standalone four-digit years
    if re.search(r'\b(19|20)\d{2}\b', normalized_text):
        logger.info("YEAR -> NO TITLE")
        return True

    # Rule 3: Use the dateutil library to robustly check for dates
    words = normalized_text.split()
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            substring = " ".join(words[i:j])
            try:
                parse_date(substring)
                logger.info("DATE PARSED")
                return True
            except (ParserError, ValueError, OverflowError, TypeError):
                pass
    

    logger.info("TITLE FOUND")
    return False
