"""
Configuration and constants for SEC financial scraper.
"""

import re
from dataclasses import dataclass
from typing import Dict, List

# ============================================================================
# USER AGENT
# ============================================================================

# SEC requires a custom User-Agent for all API requests
# Replace with your actual information
HEADERS = {'User-Agent': 'Your Name your.email@example.com'}


# ============================================================================
# REGEX PATTERNS
# ============================================================================

class Patterns:
    """Pre-compiled regex patterns for performance."""
    
    # Year patterns
    YEAR = re.compile(r'\b(19\d{2}|20\d{2})\b')
    YEAR_ONLY = re.compile(r'^(19|20)\d{2}$')
    
    # Period patterns
    FY = re.compile(
        r"\b(?:twelve\s*months|(?:fiscal\s+)?years?)\s*ended\b", 
        re.IGNORECASE
    )
    NINE_MONTHS = re.compile(
        r"\b(?:nine|9)\s*months?\s*ended\b", 
        re.IGNORECASE
    )
    SIX_MONTHS = re.compile(
        r"\b(?:six|6)\s*months?\s*ended\b", 
        re.IGNORECASE
    )
    QUARTER = re.compile(
        r"\b(?:three|3)\s*months?\s*ended|quarter(?:ly)?\s+period\b", 
        re.IGNORECASE
    )
    
    # Date parsing for fallback
    DATE = re.compile(
        r"""
        (?:(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+)?
        \b(20\d{2})\b
        """,
        re.VERBOSE | re.IGNORECASE
    )
    
    # Units pattern
    UNITS = re.compile(
        r'^\s*'
        r'(?:'
            r'\([^)]*(?:millions|thousands|billions|shares|per\s+share|dollars)[^)]*\)|'
            r'\(?\s*(?:amounts?\s+)?in\s+(?:thousands|millions|billions|shares|dollars|whole\s+dollars)'
        r')',
        re.IGNORECASE
    )
    
    # Item 8 pattern
    ITEM_8 = re.compile(r'^\s*item\s*(8|viii)[\.\s]*', re.IGNORECASE)
    FINANCIAL_STATEMENTS = re.compile(r'financial\s*statements', re.IGNORECASE)


# ============================================================================
# STATEMENT KEYWORDS
# ============================================================================

@dataclass(frozen=True)
class StatementKeywords:
    """Keywords for identifying financial statement types."""
    must_have: List[str]
    nice_to_have: List[str]
    penalties: List[str]


STATEMENT_KEYWORDS = {
    'Comprehensive Income Statement': StatementKeywords(
        must_have=['income', 'operations', 'earnings'],
        nice_to_have=['loss', 'statements', 'consolidated', 'comprehensive'],
        penalties=['tax', 'taxes', 'per share', 'note', 'schedule', 'discussion', 'other']
    ),
    'Income Statement': StatementKeywords(
        must_have=['income', 'operations', 'earnings'],
        nice_to_have=['loss', 'statements', 'consolidated'],
        penalties=['tax', 'taxes', 'per share', 'note', 'schedule', 'discussion', 'other', 'comprehensive']
    ),
    'Balance Sheet': StatementKeywords(
        must_have=['balance sheet'],
        nice_to_have=['statements', 'consolidated'],
        penalties=['tax', 'note', 'schedule', 'discussion', 'other']
    ),
    'Cash Flow Statement': StatementKeywords(
        must_have=['cash flow', 'cash flows'],
        nice_to_have=['statements', 'consolidated'],
        penalties=['note', 'schedule', 'discussion', 'other']
    ),
    'Stockholders Equity': StatementKeywords(
        must_have=["stockholders' equity", "shareholders' equity"],
        nice_to_have=['statements', 'consolidated'],
        penalties=['note', 'schedule', 'discussion', 'comprehensive', 'other']
    )
}


# ============================================================================
# MONTH ABBREVIATIONS
# ============================================================================

MONTH_ABBREV = {
    'jan': 'january', 'feb': 'february', 'mar': 'march',
    'apr': 'april', 'may': 'may', 'jun': 'june',
    'jul': 'july', 'aug': 'august', 'sep': 'september',
    'oct': 'october', 'nov': 'november', 'dec': 'december'
}


# ============================================================================
# TARGET SECTIONS
# ============================================================================

TARGET_SECTIONS = [
    'Income Statement',
    'Balance Sheet',
    'Cash Flow Statement',
    'Comprehensive Income Statement',
    'Stockholders Equity'
]


# ============================================================================
# SCRAPER CONFIGURATION
# ============================================================================

@dataclass
class ScraperConfig:
    """Configuration for the scraper."""
    
    # Metric column overrides for specific ticker+section combinations
    metric_column_overrides: Dict[str, int] = None
    
    # Maximum number of filings to process per run (None = all)
    max_filings: int = None
    
    # Enable verbose logging
    verbose: bool = False
    
    def __post_init__(self):
        if self.metric_column_overrides is None:
            self.metric_column_overrides = {}
    
    def get_metric_column(self, ticker: str, section_name: str, default: int = 0) -> int:
        """Get the metric column for a specific section and ticker."""
        key = f"{ticker}_{section_name}"
        return self.metric_column_overrides.get(key, default)
    
    def set_metric_column(self, ticker: str, section_name: str, column: int):
        """Store a metric column exception for this ticker and section."""
        key = f"{ticker}_{section_name}"
        self.metric_column_overrides[key] = column