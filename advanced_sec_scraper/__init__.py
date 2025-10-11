"""
Advanced SEC Financial Scraper - Modular Edition

A modular Python-based scraper for extracting financial data from SEC 10-K and 10-Q filings.
"""

__version__ = "2.0.0"
__author__ = "Daniel Born"

# Public API
from .config import ScraperConfig, STATEMENT_KEYWORDS, TARGET_SECTIONS
from .sec_client import SECClient, generate_fiscal_calendar
from .statement_finder import StatementFinder
from .table_processor import TableProcessor, normalize_text
from .table_header_parser import TableHeaderParser
from .data_extractor import run_scraping, ScrapingContext

__all__ = [
    # Main functions
    'run_scraping',
    
    # Configuration
    'ScraperConfig',
    'STATEMENT_KEYWORDS',
    'TARGET_SECTIONS',
    
    # Components (for advanced usage)
    'SECClient',
    'StatementFinder',
    'TableProcessor',
    'TableHeaderParser',
    'ScrapingContext',
    
    # Utilities
    'generate_fiscal_calendar',
    'normalize_text',
]