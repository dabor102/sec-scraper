import sys
import os
import re
import logging
import requests
import json
import traceback
import time
from collections import Counter
from dataclasses import dataclass
import pandas as pd
from bs4 import BeautifulSoup, Tag, NavigableString
from dateutil.parser import parse as parse_date

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Main parser classes and tools
from sec_parser.processing_engine.core import Edgar10KParser, Edgar10QParser
from sec_parser.semantic_tree.tree_builder import TreeBuilder
from sec_parser.semantic_elements.table_element.table_element import TableElement
from sec_parser.semantic_elements.title_element import TitleElement
from sec_parser.semantic_elements.top_section_title import TopSectionTitle


# ============================================================================
# CONFIGURATION
# ============================================================================

HEADERS = {'User-Agent': 'Your Name your.email@example.com'}

# Statement keywords for finding financial statements
STATEMENT_KEYWORDS = {
    'Comprehensive Income Statement': {
        'must_have': ['income', 'operations', 'earnings'],
        'nice_to_have': ['loss', 'statements', 'consolidated', 'comprehensive'],
        'penalties': ['tax', 'taxes', 'per share', 'note', 'schedule', 'discussion', 'other']
    },
    'Income Statement': {
        'must_have': ['income', 'operations', 'earnings'],
        'nice_to_have': ['loss', 'statements', 'consolidated'],
        'penalties': ['tax', 'taxes', 'per share', 'note', 'schedule', 'discussion', 'other', 'comprehensive']
    },
    'Balance Sheet': {
        'must_have': ['balance sheet'],
        'nice_to_have': ['statements', 'consolidated'],
        'penalties': ['tax', 'note', 'schedule', 'discussion','other']
    },
    'Cash Flow Statement': {
        'must_have': ['cash flow', 'cash flows'],
        'nice_to_have': ['statements', 'consolidated'],
        'penalties': ['note', 'schedule', 'discussion' ,'other']
    },
    'Stockholders Equity': {
        'must_have': ["stockholders' equity", "shareholders' equity"],
        'nice_to_have': ['statements', 'consolidated'],
        'penalties': ['note', 'schedule', 'discussion', 'comprehensive', 'other']
    }
}

MONTH_ABBREV = {
    'jan': 'january', 'feb': 'february', 'mar': 'march',
    'apr': 'april', 'may': 'may', 'jun': 'june',
    'jul': 'july', 'aug': 'august', 'sep': 'september',
    'oct': 'october', 'nov': 'november', 'dec': 'december'
}


# Ticker-specific metric column overrides (populated at runtime if needed)
METRIC_COLUMN_EXCEPTIONS = {}


# ============================================================================
# CONTEXT CLASS - Groups related scraping parameters
# ============================================================================

@dataclass
class ScrapingContext:
    """Container for scraping configuration and state."""
    fiscal_calendar: dict
    ticker: str
    form_type: str
    metric_col_exceptions: dict
    
    def get_metric_column(self, section_name: str, default: int = 0) -> int:
        """Get the metric column for a specific section and ticker."""
        key = f"{self.ticker}_{section_name}"
        return self.metric_col_exceptions.get(key, default)
    
    def set_metric_column(self, section_name: str, column: int):
        """Store a metric column exception for this ticker and section."""
        key = f"{self.ticker}_{section_name}"
        self.metric_col_exceptions[key] = column


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Sets up a logger that outputs to console (INFO) and a file (DEBUG)."""
    logger = logging.getLogger("SEC_Scraper")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    if logger.handlers:
        return logger

    file_handler = logging.FileHandler("scraper_debug.log", mode='w')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logging()

def save_semantic_tree_debug(semantic_tree: list, ticker: str, filing_date: str = None, 
                            reason: str = None, section_name: str = None):
    """
    Saves semantic tree to CSV for debugging purposes.
    
    Args:
        semantic_tree: List of semantic tree nodes
        ticker: Stock ticker symbol
        filing_date: Optional filing date for filename
        reason: Reason for saving (e.g., 'no_item8', 'no_data_extracted', 'exception')
        section_name: Optional section name if failure was section-specific
    """
    if not semantic_tree:
        logger.warning("No semantic tree to save")
        return
    
    try:
        # Create debug data
        debug_records = []
        for idx, node in enumerate(semantic_tree):
            record = {
                'Index': idx,
                'Element_Type': type(node.semantic_element).__name__ if hasattr(node, 'semantic_element') else 'Unknown',
                'Text': normalize_text(node.text) if hasattr(node, 'text') else '',
                'Text_Length': len(node.text) if hasattr(node, 'text') else 0,
                'Has_Children': len(node.children) if hasattr(node, 'children') else 0,
                'Level': getattr(node, 'level', 'N/A'),
            }
            
            # Add any additional attributes that might be useful
            if hasattr(node.semantic_element, '__dict__'):
                for key, value in node.semantic_element.__dict__.items():
                    if key not in ['text', 'html_tag'] and not key.startswith('_'):
                        record[f'Attr_{key}'] = str(value)[:100]  # Limit length
            
            debug_records.append(record)
        
        filename = f"{ticker}_semantic_tree.csv"
        
        # Save to CSV
        df = pd.DataFrame(debug_records)
        df.to_csv(filename, index=False)
        
        logger.info(f"Semantic tree saved for debugging: {filename}")
        logger.info(f"Total nodes saved: {len(debug_records)}")

        return filename
        
    except Exception as e:
        logger.error(f"Failed to save semantic tree debug file: {e}")
        logger.debug(traceback.format_exc())

# ============================================================================
# TEXT UTILITIES
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace while preserving content."""
    if not text:
        return ""
    return ' '.join(text.split())


def is_empty_or_dash(cell_value: str) -> bool:
    """Checks if a cell represents an empty/missing value."""
    if not cell_value or not cell_value.strip():
        return True
    stripped = cell_value.strip()
    dash_chars = {'-', '—', '–', '−', '―', '‐'}
    return stripped in dash_chars


# ============================================================================
# FISCAL CALENDAR
# ============================================================================

def generate_fiscal_calendar(fye_month: int) -> dict:
    """
    Generates a dynamic fiscal month-to-quarter mapping based on fiscal year-end month.
    
    Args:
        fye_month: Fiscal year-end month (1-12)
        
    Returns:
        Dict mapping month names to quarters (Q1-Q4)
    """
    if not 1 <= fye_month <= 12:
        fye_month = 12  # Default to December
        logger.warning(f"Invalid FYE month, defaulting to December")

    months = ['january', 'february', 'march', 'april', 'may', 'june',
              'july', 'august', 'september', 'october', 'november', 'december']
    
    # Reorder months so fiscal year starts at the beginning
    fiscal_months = months[fye_month:] + months[:fye_month]

    calendar = {}
    calendar.update({m: 'Q1' for m in fiscal_months[0:3]})
    calendar.update({m: 'Q2' for m in fiscal_months[3:6]})
    calendar.update({m: 'Q3' for m in fiscal_months[6:9]})
    calendar.update({m: 'Q4' for m in fiscal_months[9:12]})
    
    logger.debug(f"Generated fiscal calendar for FYE month {fye_month}: {calendar}")
    return calendar


# ============================================================================
# PERIOD PARSING
# ============================================================================

def parse_period(header_text: str, context: ScrapingContext, 
                is_snapshot: bool = False) -> str | None:
    """
    Unified period parser for both snapshot (balance sheet) and duration statements.
    Enhanced to handle month abbreviations, full date formats, and correct fiscal year assignment.
    
    Args:
        header_text: Text to parse for period information
        context: Scraping context with fiscal calendar
        is_snapshot: True for balance sheets, False for income/cash flow statements
    """
    if not header_text or not header_text.strip():
        return None
    
    # Sanitize text
    header_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', header_text)
    header_text = re.sub(r'(\d+),(\d{4})', r'\1, \2', header_text)
    
    header_lower = header_text.lower()
    
    # Extract year
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', header_text)
    if not year_match:
        return None
    year = year_match.group(1)
    
    # Try to parse as a date first using dateutil
    found_month = None
    found_quarter = None
    parsed_date = None
    
    try:
        # Try parsing the entire string as a date
        parsed_date = parse_date(header_text, fuzzy=True)
        month_num = parsed_date.month
        month_name = parsed_date.strftime('%B').lower()  # Full month name
        
        # Check if this month is in fiscal calendar
        if month_name in context.fiscal_calendar:
            found_month = month_name
            found_quarter = context.fiscal_calendar[month_name]
            logger.debug(f"Parsed date '{header_text}' -> Month: {month_name}, Quarter: {found_quarter}")
    except (ValueError, OverflowError, AttributeError):
        # If date parsing fails, try keyword matching
        pass
    
    # If date parsing didn't work, try finding month names or abbreviations
    if not found_month:
        # First try full month names
        for month in context.fiscal_calendar:
            if month in header_lower:
                found_month = month
                found_quarter = context.fiscal_calendar[month]
                break
        
        # If not found, try abbreviations
        if not found_month:
            for abbrev, full_month in MONTH_ABBREV.items():
                # Use word boundary to avoid matching "december" when looking for "dec"
                if re.search(r'\b' + abbrev + r'\b', header_lower):
                    if full_month in context.fiscal_calendar:
                        found_month = full_month
                        found_quarter = context.fiscal_calendar[full_month]
                        logger.debug(f"Found month abbreviation '{abbrev}' -> {full_month} -> {found_quarter}")
                        break
    
    # Balance sheet / snapshot periods
    if is_snapshot:
        if found_quarter:
            # Determine the correct fiscal year
            fiscal_year = determine_fiscal_year(
                calendar_year=int(year),
                month_name=found_month,
                parsed_date=parsed_date,
                context=context
            )
            return f"{found_quarter} {fiscal_year}"
        # Fallback: assume Q4/year-end if only year is present
        logger.debug(f"No month found in '{header_text}'. Assuming Q4 for balance sheet.")
        return f"Q4 {year}"
    
    # Duration periods (income statement, cash flow, etc.)
    if 'twelve months' in header_lower or 'year ended' in header_lower or 'fiscal year' in header_lower:
        return f"FY {year}"
    elif 'nine months' in header_lower:
        return f"Q1-Q3 {year}"
    elif 'six months' in header_lower:
        return f"Q1-Q2 {year}"
    elif 'three months' in header_lower or 'quarter' in header_lower:
        return f"{found_quarter} {year}" if found_quarter else None
    elif found_quarter:
        # Fallback for headers with only month/date
        return f"{found_quarter} {year}"
    
    return None

def determine_fiscal_year(calendar_year: int, month_name: str, 
                          parsed_date, context: ScrapingContext) -> int:
    """
    Determines the fiscal year for a given date based on the company's fiscal year end.
    
    For balance sheets, the fiscal year is the year in which the fiscal period ENDS.
    
    Args:
        calendar_year: The calendar year from the date (e.g., 2025)
        month_name: The month name (e.g., 'january')
        parsed_date: The parsed datetime object (if available)
        context: Scraping context with fiscal calendar
        
    Returns:
        The fiscal year (e.g., 2024 for a balance sheet dated January 31, 2025 with FYE in January)
    """
    # Get the fiscal year-end month number (1-12)
    # We need to reverse-engineer this from the fiscal calendar
    # The FYE month is the last month in Q4
    
    fye_month_name = None
    for month, quarter in context.fiscal_calendar.items():
        if quarter == 'Q4':
            fye_month_name = month  # Keep updating, last one will be the FYE
    
    if not fye_month_name:
        logger.warning("Could not determine FYE month from fiscal calendar")
        return calendar_year
    
    # Convert month name to number
    month_to_num = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    fye_month_num = month_to_num.get(fye_month_name)
    
    if parsed_date:
        date_month_num = parsed_date.month
    else:
        # Fall back to month_name if no parsed_date
        date_month_num = month_to_num.get(month_name)
    
    if not fye_month_num or not date_month_num:
        logger.warning(f"Could not determine month numbers for FYE or date")
        return calendar_year
    
    # Rule: If the date month is <= FYE month, it's the end of the previous fiscal year
    # Example: FYE in January (1), date in January 2025 → FY2024
    # Example: FYE in January (1), date in July 2025 → FY2025
    
    if date_month_num <= fye_month_num:
        fiscal_year = calendar_year - 1
        logger.debug(f"Date month {date_month_num} <= FYE month {fye_month_num}: "
                    f"Fiscal year = {fiscal_year} (calendar year - 1)")
    else:
        fiscal_year = calendar_year
        logger.debug(f"Date month {date_month_num} > FYE month {fye_month_num}: "
                    f"Fiscal year = {fiscal_year} (calendar year)")
    
    return fiscal_year



def correct_fiscal_periods(periods: list[str]) -> list[str]:
    """
    Corrects fiscal quarter periods based on cumulative period context.
    Logic: Q1-Q3 implies Q3, Q1-Q2 implies Q2, standalone quarter implies Q1.
    """
    if not periods:
        return []

    year_anchors = {}
    single_quarter_pattern = re.compile(r'^(Q\d) (\d{4})$')
    
    # Find cumulative period anchors
    for period in periods:
        if 'Q1-Q3' in period:
            year = period.split()[-1]
            year_anchors[year] = 'Q3'
        elif 'Q1-Q2' in period:
            year = period.split()[-1]
            year_anchors[year] = 'Q2'
    
    # If no anchors and only one single-quarter period, assume Q1
    if not year_anchors:
        single_quarters = [p for p in periods if single_quarter_pattern.match(p)]
        if len(single_quarters) == 1:
            logger.info("Only one single-quarter period found. Assuming it's Fiscal Q1.")
            q1_period = single_quarters[0]
            year = q1_period.split()[-1]
            return [f'Q1 {year}' if p == q1_period else p for p in periods]
        return periods

    # Apply corrections using anchors
    corrected_periods = []
    for period in periods:
        match = single_quarter_pattern.match(period)
        if match:
            year = match.group(2)
            if year in year_anchors:
                corrected_period = f"{year_anchors[year]} {year}"
                logger.info(f"Correcting '{period}' to '{corrected_period}' based on cumulative anchor.")
                corrected_periods.append(corrected_period)
            else:
                corrected_periods.append(period)
        else:
            corrected_periods.append(period)
            
    return corrected_periods


# ============================================================================
# HEADER PARSING - Consolidated and Simplified
# ============================================================================

def classify_cell_text(text: str) -> set:
    """Analyzes table cell text and returns information types it contains."""
    if not text or not text.strip():
        return set()

    classifications = set()
    text_lower = text.lower()
    text_cleaned = text.strip()

    # Check for duration keywords
    duration_keywords = ['months ended', 'year ended', 'fiscal year', 'quarter']
    if any(keyword in text_lower for keyword in duration_keywords):
        classifications.add('DURATION')

    # Check for standalone year
    year_pattern = re.compile(r'^\s*(19|20)\d{2}\s*$')
    if year_pattern.match(text_cleaned):
        classifications.add('YEAR')
        return classifications

    # Try to parse date
    try:
        parse_date(text_cleaned, fuzzy=True)
        classifications.add('DATE')
        if re.search(r'\b(19|20)\d{2}\b', text_cleaned):
            classifications.add('YEAR')
    except (ValueError, OverflowError):
        if re.search(r'\b(19|20)\d{2}\b', text_cleaned):
            classifications.add('YEAR')
            
    return classifications


def parse_header(header_rows: list[list[str]], metric_col: int, 
                context: ScrapingContext, is_snapshot: bool = False) -> list[str]:
    """
    Unified header parser that handles various layouts.
    
    Improved to better handle split headers (description in one row, years in another).
    """
    if not header_rows:
        return []
    
    # Try horizontal layout first
    periods = _try_horizontal_header(header_rows, metric_col, context, is_snapshot)
    if periods:
        return periods
    
    # Try split header layout (common pattern: description row + years row)
    periods = _try_split_header(header_rows, metric_col, context, is_snapshot)
    if periods:
        return periods
    
    # Try broadcasting layout
    periods = _try_broadcasting_header(header_rows, metric_col, context, is_snapshot)
    if periods:
        return periods
    
    # Fallback to simple vertical layout
    periods = _try_vertical_header(header_rows, metric_col, context, is_snapshot)
    return periods
   
def _try_split_header(header_rows: list[list[str]], metric_col: int,
                     context: ScrapingContext, is_snapshot: bool) -> list[str]:
    """
    Handle split headers where description and years are in separate rows.

    Common pattern:
    Row N:   [empty/units] | "Fiscal Years Ended January 31," (colspan)
    Row N+1: [units/empty] | "2023" | "2022" | "2021"
    
    Improved to match descriptions to years based on column positions.
    """
    logger.debug("Attempting split header format")
    
    # Patterns to identify description rows
    description_patterns = [
        re.compile(r'(?:fiscal\s+)?(?:years?|months?|quarters?)\s+ended', re.IGNORECASE),
        re.compile(r'(?:three|six|nine|twelve)\s+months?\s+ended', re.IGNORECASE),
        re.compile(r'as\s+of', re.IGNORECASE),
    ]
    
    year_pattern = re.compile(r'\b(19|20)\d{2}\b')
    
    for desc_row_idx in range(len(header_rows) - 1):
        desc_row = header_rows[desc_row_idx]
        
        # Build a map of column positions to description text
        desc_map = {}  # {col_idx: description_text}
        
        for col_idx, cell in enumerate(desc_row):
            if col_idx <= metric_col:
                continue
            
            cell_text = cell.strip()
            if not cell_text:
                continue
            
            # Check if this cell has a description
            has_description = any(pattern.search(cell_text.lower()) 
                                for pattern in description_patterns)
            
            if has_description:
                desc_map[col_idx] = cell_text
        
        if not desc_map:
            continue
        
        logger.debug(f"Found description row {desc_row_idx} with {len(desc_map)} descriptions at columns: {list(desc_map.keys())}")
        
        # Look at next few rows for years
        for year_row_idx in range(desc_row_idx + 1, min(desc_row_idx + 4, len(header_rows))):
            year_row = header_rows[year_row_idx]
            
            # Build a map of column positions to years
            year_map = {}  # {col_idx: year_text}
            
            for col_idx, cell in enumerate(year_row):
                if col_idx <= metric_col:
                    continue
                
                cell_text = cell.strip()
                if cell_text and year_pattern.search(cell_text):
                    year_map[col_idx] = cell_text
            
            if not year_map:
                continue
            
            logger.debug(f"Found year row {year_row_idx} with {len(year_map)} years at columns: {list(year_map.keys())}")
            
            # Match each year to its corresponding description based on column proximity
            periods = []
            year_columns = sorted(year_map.keys())
            
            for year_col in year_columns:
                year_text = year_map[year_col]
                
                # Find the description that "covers" this year column
                # A description "covers" a year if:
                # 1. The description column is before or at the year column
                # 2. It's the closest description to the left of the year
                
                best_desc = None
                best_desc_col = -1
                
                for desc_col, desc_text in desc_map.items():
                    # The description should be at or before the year column
                    # But we need to account for colspan expansion
                    # Generally, we want the rightmost description that's still <= year_col
                    if desc_col <= year_col and desc_col > best_desc_col:
                        best_desc = desc_text
                        best_desc_col = desc_col
                
                # If no description to the left, try finding one in a nearby column
                # (handles cases where colspan shifted positions slightly)
                if best_desc is None and desc_map:
                    # Find description with minimum distance to year column
                    min_distance = float('inf')
                    for desc_col, desc_text in desc_map.items():
                        distance = abs(desc_col - year_col)
                        if distance < min_distance:
                            min_distance = distance
                            best_desc = desc_text
                            best_desc_col = desc_col
                
                if best_desc:
                    combined_text = f"{best_desc} {year_text}"
                    period = parse_period(combined_text, context, is_snapshot)
                    
                    if period and period not in periods:
                        periods.append(period)
                        logger.debug(f"Matched year at col {year_col} to description at col {best_desc_col}: '{combined_text}' -> {period}")
                else:
                    logger.debug(f"Could not find matching description for year at column {year_col}")
            
            if periods:
                logger.info(f"Successfully parsed split header: {periods}")
                return periods
    
    return []

def _try_horizontal_header(header_rows: list[list[str]], metric_col: int,
                           context: ScrapingContext, is_snapshot: bool) -> list[str]:
    """
    Try to parse as horizontal header (description on left, dates/years on right).
    Enhanced to handle both standalone years and full dates.
    """
    # Match standalone years OR full dates with years
    year_pattern = re.compile(r'\b(19|20)\d{2}\b')
    
    logger.debug("Attempting horizontal header format")
    for row in header_rows:
        if not any(cell.strip() for cell in row):
            continue

        base_desc_cells = [cell.strip() for cell in row[:metric_col + 1] if cell.strip()]
        data_cells = row[metric_col + 1:]
        
        # Need description and at least one cell with a year
        year_containing_cells = [cell for cell in data_cells if year_pattern.search(cell)]
        
        if not base_desc_cells or not year_containing_cells:
            continue

        base_description = normalize_text(' '.join(base_desc_cells))
        periods = []
        
        # Process only non-empty cells that contain years
        for cell in year_containing_cells:
            cell_stripped = cell.strip()
            if not cell_stripped:
                continue
                
            combined_text = f"{base_description} {cell_stripped}"
            period = parse_period(combined_text, context, is_snapshot)
            if period and period not in periods:
                periods.append(period)
        
        if periods:
            logger.info(f"Successfully parsed horizontal header: {periods}")
            return periods
    
    return []

def _try_broadcasting_header(header_rows: list[list[str]], metric_col: int,
                             context: ScrapingContext, is_snapshot: bool) -> list[str]:
    """
    Try broadcasting approach for layered headers.
    
    Improved to handle empty cells from colspan better and to look across ALL rows
    for context clues.
    """
    logger.debug("Attempting broadcasting header format")
    
    # Collect all meaningful cells with their column positions
    processed_rows = []
    for row_idx, row in enumerate(header_rows):
        cells = row[metric_col + 1:]
        meaningful_cells = []
        
        for col_idx, cell in enumerate(cells):
            cell_text = cell.strip()
            if cell_text:
                meaningful_cells.append({
                    'text': cell_text,
                    'position': col_idx,
                    'classifications': classify_cell_text(cell_text),
                    'row': row_idx
                })
        
        if meaningful_cells:
            processed_rows.append(meaningful_cells)
    
    if not processed_rows:
        return []

    # Find the row with the most items - likely the base period row
    base_row = max(processed_rows, key=len)
    num_periods = len(base_row)
    
    # If base row only has 1 item, this probably isn't the right approach
    if num_periods < 2:
        logger.debug(f"Broadcasting found only {num_periods} period(s), skipping")
        return []
    
    logger.debug(f"Broadcasting: detected {num_periods} periods from base row")

    # Map each position in base row to its column
    position_to_column = {i: cell['position'] for i, cell in enumerate(base_row)}
    
    # Initialize context for each period
    resolved_contexts = [{'duration': [], 'date': [], 'year': []} for _ in range(num_periods)]

    # Broadcast information from ALL rows
    for row in processed_rows:
        # Try to map cells to period positions based on column alignment
        for cell_info in row:
            col_pos = cell_info['position']
            text = cell_info['text']
            classifications = cell_info['classifications']
            
            # Find which period this cell belongs to based on column proximity
            best_period_idx = None
            min_distance = float('inf')
            
            for period_idx, base_col in position_to_column.items():
                distance = abs(col_pos - base_col)
                if distance < min_distance:
                    min_distance = distance
                    best_period_idx = period_idx
            
            # Allow some tolerance for column alignment (within 2 columns)
            if best_period_idx is not None and min_distance <= 2:
                if 'DURATION' in classifications:
                    resolved_contexts[best_period_idx]['duration'].append(text)
                if 'DATE' in classifications:
                    resolved_contexts[best_period_idx]['date'].append(text)
                if 'YEAR' in classifications:
                    resolved_contexts[best_period_idx]['year'].append(text)

    # Reconstruct period strings
    ordered_periods = []
    for idx, ctx in enumerate(resolved_contexts):
        duration_text = normalize_text(' '.join(dict.fromkeys(ctx['duration'])))
        date_text = normalize_text(' '.join(dict.fromkeys(ctx['date'])))
        year_text = normalize_text(' '.join(dict.fromkeys(ctx['year'])))
        
        combined_text = f"{duration_text} {date_text} {year_text}".strip()
        
        # If we only have a year, try to get description from other contexts
        if not combined_text or (year_text and not duration_text and not date_text):
            # Use common description from first context if available
            if resolved_contexts[0]['duration']:
                common_desc = resolved_contexts[0]['duration'][0]
                combined_text = f"{common_desc} {year_text}".strip()
        
        period = parse_period(combined_text, context, is_snapshot)
        
        if period:
            ordered_periods.append(period)
            logger.debug(f"Period {idx}: '{combined_text}' -> {period}")

    if ordered_periods and len(ordered_periods) == num_periods:
        logger.info(f"Successfully parsed {len(ordered_periods)} periods via broadcasting")
        return ordered_periods
    
    return []

def _try_vertical_header(header_rows: list[list[str]], metric_col: int,
                        context: ScrapingContext, is_snapshot: bool) -> list[str]:
    """
    Simple vertical header parsing with improved colspan handling.
    
    Strategy:
    1. Identify columns that have actual date/period content
    2. Skip empty cells created by colspan expansion
    3. Parse each unique column position
    """
    if not header_rows:
        return []
    
    max_cols = max(len(row) for row in header_rows) if header_rows else 0
    if max_cols <= metric_col + 1:
        return []
    
    # Build a map of which columns have actual content across all header rows
    column_content = {}  # col_idx -> list of texts from different rows
    
    for row in header_rows:
        for col_idx in range(metric_col + 1, len(row)):
            cell_text = row[col_idx].strip()
            if cell_text:  # Only record non-empty cells
                if col_idx not in column_content:
                    column_content[col_idx] = []
                # Avoid duplicates in the same column
                if not column_content[col_idx] or column_content[col_idx][-1] != cell_text:
                    column_content[col_idx].append(cell_text)
    
    if not column_content:
        return []
    
    # Group columns that likely represent the same period
    # Strategy: columns close together with similar content patterns are likely the same period
    period_groups = []
    sorted_cols = sorted(column_content.keys())
    
    current_group = []
    last_col = -1
    
    for col_idx in sorted_cols:
        # If this column is far from the last one, start a new group
        # "Far" means more than 3 columns apart (accounts for typical colspan patterns)
        if last_col != -1 and col_idx - last_col > 3:
            if current_group:
                period_groups.append(current_group)
            current_group = []
        
        current_group.append(col_idx)
        last_col = col_idx
    
    if current_group:
        period_groups.append(current_group)
    
    # Parse periods from each group
    periods = []
    for group in period_groups:
        # Collect all text from columns in this group
        group_texts = []
        for col_idx in group:
            group_texts.extend(column_content[col_idx])
        
        # Remove duplicates while preserving order
        unique_texts = list(dict.fromkeys(group_texts))
        combined_text = ' '.join(unique_texts)
        
        period = parse_period(combined_text, context, is_snapshot)
        if period and (not periods or period != periods[-1]):
            periods.append(period)
            logger.debug(f"Parsed period from columns {group}: '{combined_text}' -> {period}")
    
    return periods



# ============================================================================
# TABLE UTILITIES
# ============================================================================

def extract_table_from_semantic_node(title_node) -> tuple[Tag, dict] | None:
    """
    Extract table directly using semantic tree references.
    
    Returns:
        (bs4_table, metadata) or None
    """
    metadata = {
        'source': None,
        'title_bs4': None,
        'is_extracted_title': False,
        'title_row_index': None,
    }
    
    # Get the actual HTML element from semantic tree
    title_element = title_node.semantic_element
    title_bs4 = title_element.html_tag._bs4
    metadata['title_bs4'] = title_bs4
    
    logger.debug(f"Title HTML tag: <{title_bs4.name}>")
    
    # STRATEGY 1: Check if title is part of CompositeSemanticElement
    parent_node = title_node.parent if hasattr(title_node, 'parent') else None
    
    if parent_node:
        from sec_parser.semantic_elements.composite_semantic_element import CompositeSemanticElement
        
        if isinstance(parent_node.semantic_element, CompositeSemanticElement):
            logger.info("Title is in CompositeSemanticElement - looking for TableElement sibling")
            
            for sibling_node in parent_node.children:
                if isinstance(sibling_node.semantic_element, TableElement):
                    table_bs4 = sibling_node.semantic_element.html_tag._bs4
                    
                    # Check if NOT a wrapper (i.e., it IS a data table)
                    if not _is_title_wrapper_table(table_bs4, title_bs4):
                        logger.info("✓ Found associated table via CompositeSemanticElement")
                        metadata['source'] = 'composite'
                        metadata['is_extracted_title'] = True
                        return table_bs4, metadata
                    else:
                        logger.debug("Sibling table is just a wrapper, not a data table")
    
    # STRATEGY 2: Check if title itself is inside a table
    parent_table = title_bs4.find_parent('table')
    
    if parent_table:
        # Check if NOT a wrapper (i.e., it IS a data table)
        # Pass title_bs4 so _is_title_wrapper_table knows to check for it
        if not _is_title_wrapper_table(parent_table, title_bs4):
            logger.info("✓ Title is inside a data table")
            
            # Find which row the title is in
            title_row = title_bs4.find_parent('tr')
            if title_row:
                all_rows = parent_table.find_all('tr')
                try:
                    title_row_index = all_rows.index(title_row)
                    metadata['title_row_index'] = title_row_index
                    logger.debug(f"Title is in row {title_row_index}")
                except ValueError:
                    logger.warning("Could not determine title row index")
            
            metadata['source'] = 'parent_table'
            return parent_table, metadata
        else:
            logger.debug("Parent table is just a wrapper, looking for next table")
            # Fall through to strategy 3
    
    # STRATEGY 3: Look for next table after title
    logger.info("Looking for next table after title element")
    next_table = find_next_table(title_bs4)
    
    if next_table:
        logger.info("✓ Found next table after title")
        metadata['source'] = 'next_table'
        return next_table, metadata
    
    logger.warning("Could not find table for title")
    return None


def build_grid_from_table(table: Tag) -> list[list[str]]:
    """Builds accurate 2D list from BeautifulSoup table tag."""
    grid = []
    for r, row_tag in enumerate(table.find_all('tr')):
        while len(grid) <= r:
            grid.append([])
        for cell_tag in row_tag.find_all(['th', 'td']):
            c = 0
            while c < len(grid[r]) and grid[r][c] is not None:
                c += 1
            while c >= len(grid[r]):
                grid[r].append(None)
            
            colspan = int(cell_tag.get('colspan', 1))
            rowspan = int(cell_tag.get('rowspan', 1))

            # Remove superscript tags (footnote markers)
            for sup_tag in cell_tag.find_all('sup'):
                sup_tag.decompose()
            
            cell_text = normalize_text(cell_tag.get_text(strip=True))
            for i in range(rowspan):
                for j in range(colspan):
                    while len(grid) <= r + i:
                        grid.append([])
                    while len(grid[r+i]) <= c + j:
                        grid[r+i].append(None)
                    grid[r+i][c+j] = cell_text if i == 0 and j == 0 else ""
    
    if grid:
        max_cols = max(len(row) for row in grid)
        for row in grid:
            while len(row) < max_cols:
                row.append(None)
    
    # Convert None to empty string and remove currency symbols
    currency_symbols = {'$', '€', '£', '¥', '₹'}
    clean_grid = []
    for row in grid:
        clean_row = []
        for cell in row:
            if cell is None:
                clean_row.append('')
            elif cell.strip() in currency_symbols:
                clean_row.append('')
            else:
                clean_row.append(cell)
        clean_grid.append(clean_row)
    
    return clean_grid

def split_table_into_header_and_body(grid: list[list[str]], 
                                     metric_col: int,
                                     title_row_index: int = None) -> tuple[list[list[str]], int]:
    """
    Splits table grid into header and body sections.
    
    Args:
        grid: 2D list representing the table
        metric_col: Column index for metric names
        title_row_index: Optional - if provided, skip this row (it's the title)
    
    Returns:
        (header_rows, body_start_index)
    """
    if not grid:
        return [], 0

    header_keywords = {'year', 'ended', 'month', 'quarter', 'period', 'fiscal'}
    
    month_names = {
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'may', 'jun', 
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    }
    
    units_pattern = re.compile(
        r'^\s*'
        r'(?:'
            r'\([^)]*(?:millions|thousands|billions|shares|per\s+share|dollars)[^)]*\)|'
            r'\(?\s*(?:amounts?\s+)?in\s+(?:thousands|millions|billions|shares|dollars|whole\s+dollars)'
        r')',
        re.IGNORECASE
    )
    
    year_pattern = re.compile(r'\b(19|20)\d{2}\b')

    for row_idx, row in enumerate(grid):
        # CRITICAL: Skip the title row if we know where it is
        if title_row_index is not None and row_idx == title_row_index:
            logger.debug(f"Skipping row {row_idx} (known title row)")
            continue
        
        if row_idx > 15:  # Safety break
            break
        
        if metric_col >= len(row):
            continue

        metric_text = normalize_text(row[metric_col])
        metric_text_lower = metric_text.lower()

        # Empty metric cell is part of header
        if not metric_text:
            continue

        # Check if this looks like a title row (even if title_row_index not provided)
        # Title indicators: "consolidated statement", "balance sheet", etc.
        title_indicators = [
            'consolidated statement',
            'balance sheet',
            'statement of income',
            'statement of operations',
            'cash flow',
            'comprehensive income',
            'stockholders equity',
            'shareholders equity'
        ]
        
        if any(indicator in metric_text_lower for indicator in title_indicators):
            logger.debug(f"Row {row_idx} appears to be title: '{metric_text}'")
            continue

        # Date/period keywords indicate header
        if any(keyword in metric_text_lower for keyword in header_keywords):
            continue
            
        # Unit declarations indicate header
        if units_pattern.match(metric_text):
            logger.debug(f"Row {row_idx} identified as header (units): '{metric_text}'")
            continue
        
        # Standalone years in metric column indicate header
        year_only_pattern = re.compile(r'^(19|20)\d{2}$')
        if year_only_pattern.match(metric_text.strip()):
            logger.debug(f"Row {row_idx} identified as header (year): '{metric_text}'")
            continue
        
        # Check if metric text is a month name or contains a month
        if any(month in metric_text_lower for month in month_names):
            data_cells = row[metric_col + 1:]
            has_years = any(year_pattern.search(cell) for cell in data_cells if cell.strip())
            
            if has_years:
                logger.debug(f"Row {row_idx} identified as header (date with years): '{metric_text}'")
                continue
            else:
                if year_pattern.search(metric_text):
                    logger.debug(f"Row {row_idx} identified as header (complete date): '{metric_text}'")
                    continue
        
        # Check if this row has only years in data columns
        data_cells = row[metric_col + 1:]
        non_empty_data_cells = [cell.strip() for cell in data_cells if cell.strip()]
        
        if non_empty_data_cells:
            all_years = all(year_only_pattern.match(cell) for cell in non_empty_data_cells)
            
            if all_years:
                logger.debug(f"Row {row_idx} identified as header (row with only years in data): '{metric_text}'")
                continue

        # This row has text but no header indicators - it's the body start
        logger.info(f"Body starts at row {row_idx}: '{metric_text}'")
        return grid[:row_idx], row_idx

    logger.warning("Could not determine header/body split. Defaulting to row 0.")
    return [], 0


def extract_units_and_currency(text: str, check_symbols: bool = False) -> tuple[str, str]:
    """Extracts currency and unit information from text."""
    if not text:
        return "", ""
    
    text_lower = text.lower()
    currency = ""
    
    if check_symbols:
        if '$' in text:
            currency = "USD"
        elif '€' in text:
            currency = "EUR"
    
    if not currency:
        if 'u.s. dollars' in text_lower or 'usd' in text_lower:
            currency = "USD"
        elif 'dollars' in text_lower:
            currency = "USD"
    
    units = ""
    if 'in millions' in text_lower:
        units = "millions"
    elif 'in thousands' in text_lower:
        units = "thousands"
    elif 'in billions' in text_lower:
        units = "billions"
    
    return currency, units


def auto_detect_units_and_currency(grid: list[list[str]], 
                                   data_table: Tag) -> tuple[str, str]:
    """Automatically detects currency and units from table or surrounding text."""
    # Check table content
    table_text = normalize_text(" ".join(cell for row in grid[:15] for cell in row if cell))
    currency, units = extract_units_and_currency(table_text, check_symbols=True)
    
    if currency and units:
        return currency, units
    
    # Look around table for context
    outside_texts = []
    for element in data_table.find_all_previous(limit=10):
        if element.string and element.string.strip() and len(element.string.strip()) > 5:
            outside_texts.append(element.string.strip())
    
    unique_texts = list(dict.fromkeys(outside_texts))
    outside_text = normalize_text(" ".join(unique_texts))
    outside_currency, outside_units = extract_units_and_currency(outside_text, check_symbols=False)
    
    final_currency = currency or outside_currency
    final_units = units or outside_units
    
    logger.debug(f"Auto-detection: Currency='{final_currency}', Units='{final_units}'")
    return final_currency, final_units


# ============================================================================
# VALUE EXTRACTION
# ============================================================================

def parse_simple_value(text: str) -> float | None:
    """Parse numeric value from text."""
    if not isinstance(text, str):
        return None
    
    text = text.strip()
    for symbol in ['$', '€', '£', '¥', '₹', ',']:
        text = text.replace(symbol, '')
    
    if is_empty_or_dash(text):
        return None
    
    is_negative = text.startswith('(') and text.endswith(')')
    if is_negative:
        text = text[1:-1]
    
    try:
        value = float(text)
        return -value if is_negative else value
    except ValueError:
        return None


def extract_ordered_values_from_row(row: list[str], metric_col: int) -> list[float]:
    """Extracts numeric values in order from row, handling split negatives."""
    if len(row) <= metric_col:
        return []
    
    cells = row[metric_col + 1:]
    values = []
    i = 0
    
    while i < len(cells):
        cell = cells[i].strip()
        
        # Skip empty or currency-only cells
        if not cell or cell in ('$', '€', '£', '¥', '₹') or is_empty_or_dash(cell):
            i += 1
            continue
        
        # Handle split negatives like "(" in one cell, "123)" in another
        if cell.startswith('(') and not cell.endswith(')'):
            accumulated = cell[1:]
            j = i + 1
            while j < len(cells):
                next_cell = cells[j].strip()
                if not next_cell:
                    j += 1
                    continue
                accumulated += next_cell
                if ')' in next_cell:
                    accumulated = accumulated.replace(')', '').replace(',', '').strip()
                    try:
                        values.append(-float(accumulated))
                    except ValueError:
                        pass
                    i = j + 1
                    break
                j += 1
            else:
                i += 1
        else:
            val = parse_simple_value(cell)
            if val is not None:
                values.append(val)
            i += 1
    
    return values


# ============================================================================
# SEC EDGAR INTERACTION
# ============================================================================

def get_filing_urls(ticker: str, form_type: str = '10-Q') -> tuple[list, str | None]:
    """Fetches filing URLs and metadata for a ticker, handling pagination."""
    # Get CIK
    try:
        tickers_resp = requests.get('https://www.sec.gov/files/company_tickers.json', 
                                   headers=HEADERS)
        tickers_resp.raise_for_status()
        tickers_data = tickers_resp.json()
        cik = next((str(v['cik_str']).zfill(10) 
                   for k, v in tickers_data.items() 
                   if v['ticker'] == ticker.upper()), None)
        if not cik:
            logger.error(f"CIK not found for ticker: {ticker}")
            return [], None
    except Exception as e:
        logger.error(f"Error fetching CIK mapping: {e}")
        return [], None
    
    try:
        # Fetch main submissions index
        base_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        logger.info(f"Fetching submissions from {base_url}")
        edgar_resp = requests.get(base_url, headers=HEADERS)
        edgar_resp.raise_for_status()
        submissions_data = edgar_resp.json()

        all_filings_data = []
        
        # Extract fiscal year end
        fiscal_year_end = submissions_data.get('fiscalYearEnd')
        if fiscal_year_end:
            logger.info(f"Found Fiscal Year End for {ticker}: {fiscal_year_end}")
        else:
            logger.warning(f"Could not find Fiscal Year End for {ticker}")

        # Process paginated archive files
        paginated_files = submissions_data.get('filings', {}).get('files', [])
        for file_meta in paginated_files:
            archive_url = f"https://data.sec.gov/submissions/{file_meta['name']}"
            logger.debug(f"Fetching archive: {archive_url}")
            archive_resp = requests.get(archive_url, headers=HEADERS)
            archive_resp.raise_for_status()
            archive_data = archive_resp.json()

            # Convert dict of lists to list of dicts
            keys = list(archive_data.keys())
            if keys:
                num_filings = len(archive_data[keys[0]])
                for i in range(num_filings):
                    filing_dict = {key: archive_data[key][i] for key in keys}
                    all_filings_data.append(filing_dict)

        # Process recent filings
        recent_data = submissions_data.get('filings', {}).get('recent', {})
        if recent_data:
            keys = list(recent_data.keys())
            if keys:
                num_recent = len(recent_data[keys[0]])
                for i in range(num_recent):
                    filing_dict = {key: recent_data[key][i] for key in keys}
                    all_filings_data.append(filing_dict)

        # Filter for desired form type
        filings = []
        unique_accession_numbers = set()
        for filing_data in all_filings_data:
            accession_number = filing_data.get('accessionNumber')
            if not accession_number or accession_number in unique_accession_numbers:
                continue
            
            if filing_data.get('form') == form_type:
                filings.append({
                    'Filing Date': filing_data.get('filingDate'),
                    'Report Period End Date': filing_data.get('reportDate'),
                    'URL': f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number.replace('-', '')}/{filing_data.get('primaryDocument', '')}"
                })
                unique_accession_numbers.add(accession_number)

        logger.info(f"Found {len(filings)} '{form_type}' filings for {ticker}")
        filings.sort(key=lambda x: x['Filing Date'], reverse=True)
        return filings, fiscal_year_end
        
    except Exception as e:
        logger.error(f"Failed to fetch filings: {e}")
        logger.debug(traceback.format_exc())
        return [], None


def get_soup(url: str) -> BeautifulSoup | None:
    """Downloads content from URL and returns BeautifulSoup object."""
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return None


# ============================================================================
# SEMANTIC TREE PARSING
# ============================================================================

def parse_document_with_semantic_tree(html_content: str, 
                                     form_type: str) -> tuple[list | None, list | None]:
    """Parse document using semantic parser."""
    try:
        logger.info(f"Parsing with semantic parser for {form_type}")

        if form_type == '10-K':
            parser = Edgar10KParser()
        elif form_type == '10-Q':
            parser = Edgar10QParser()
        else:
            logger.error(f"Unsupported form type: {form_type}")
            return None, None
        
        elements = parser.parse(html_content)
        logger.debug(f"Parsed {len(elements)} semantic elements")
        
        # Build semantic tree
        tree = TreeBuilder().build(elements)
        semantic_tree_nodes = list(tree.nodes)
        
        logger.info(f"Built semantic tree with {len(semantic_tree_nodes)} nodes")
        
        return semantic_tree_nodes, elements
    except Exception as e:
        logger.error(f"Failed to parse document: {e}")
        logger.debug(traceback.format_exc())
        return None, None

def find_item8_node(tree: list) -> tuple[int, TopSectionTitle | TitleElement | None]:
    """Finds best 'Item 8' node in semantic tree."""
    item_8_pattern = re.compile(r'^\s*item\s*(8|viii)[\.\s]*', re.IGNORECASE)
    financial_statements_pattern = re.compile(r'financial\s*statements', re.IGNORECASE)

    candidates = []
    for i, node in enumerate(tree):
        if isinstance(node.semantic_element, (TopSectionTitle, TitleElement)):
            node_text = normalize_text(node.text)
            if item_8_pattern.search(node_text):
                # Look ahead to confirm it's about financial statements
                look_ahead_text = " ".join(
                    normalize_text(next_node.text) for next_node in tree[i+1:i+6]
                )
                is_confirmed = bool(financial_statements_pattern.search(look_ahead_text))
                
                candidates.append({
                    "index": i,
                    "node": node,
                    "is_confirmed": is_confirmed,
                    "is_top_section": isinstance(node.semantic_element, TopSectionTitle)
                })

    if not candidates:
        logger.warning("Could not find 'Item 8' node")
        return -1, None

    # Sort by: confirmed first, TopSectionTitle first, last occurrence first
    candidates.sort(key=lambda c: (
        0 if c['is_confirmed'] else 1,
        0 if c['is_top_section'] else 1,
        -c['index']
    ))
    
    best = candidates[0]
    logger.info(f"Found 'Item 8' at index {best['index']} "
               f"(Confirmed: {best['is_confirmed']})")
    
    return best['index'], best['node']


def find_statement_titles(tree: list, keywords: dict) -> list:
    """Find TitleElements matching statement keywords."""
    matching_titles = []
    
    logger.debug(f"Searching for must-have keywords: {keywords['must_have']}")
    logger.debug(f"Nice-to-have keywords: {keywords['nice_to_have']}")
    
    for node in tree:
        if isinstance(node.semantic_element, TitleElement):
            title_text = normalize_text(node.text)
            title_lower = title_text.lower()
            
            # CRITICAL: Must have at least one must-have keyword
            has_required = any(kw in title_lower for kw in keywords['must_have'])
            
            if not has_required:
                continue  # Skip titles without required keywords
            
            # Also check for nice-to-have keywords
            has_nice_to_have = any(kw in title_lower for kw in keywords['nice_to_have'])
            
            if has_required or has_nice_to_have:
                logger.info(f"Found matching title: '{title_text}'")
                matching_titles.append({
                    'text': title_text,
                    'node': node
                })
    
    return matching_titles

def find_text_in_html(soup: BeautifulSoup, target_text: str) -> Tag | None:
    """
    Finds an element by exact or 'starts with' match, excluding tables.
    Handles cases where text is split across multiple child elements.
    """
    normalized_target = normalize_text(target_text)
    spaceless_target = normalized_target.replace(" ", "")
    
    if not spaceless_target:
        return None
    
    # Tags that typically contain titles as direct text
    inline_tags = ['span', 'b', 'strong', 'i', 'em', 'font']
    container_tags = ['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    
    exact_matches = []
    starts_with_matches = []
    parent_matches = []  # For combined text from children

    # First pass: try individual elements
    for element in soup.find_all(inline_tags + container_tags):
        # Skip elements inside tables
        if element.find_parent('table'):
            continue

        element_text = normalize_text(element.get_text("", strip=True))
        spaceless_element_text = element_text.replace(" ", "")

        # Skip very long blocks of text
        if len(spaceless_element_text) > len(spaceless_target) * 10 and len(spaceless_element_text) > 300:
            continue

        if spaceless_element_text == spaceless_target:
            exact_matches.append(element)
        elif spaceless_element_text.startswith(spaceless_target):
            starts_with_matches.append(element)
    
    # Return if we found direct matches
    if exact_matches:
        logger.info(f"Found exact match in '{exact_matches[0].name}' tag")
        return exact_matches[0]
    
    if starts_with_matches:
        logger.info(f"Found 'starts with' match in '{starts_with_matches[0].name}' tag")
        return starts_with_matches[0]
    
    # Second pass: try combining text from sibling elements within a parent
    logger.debug("No direct match found, trying to combine text from child elements")
    
    for element in soup.find_all(container_tags):
        # Skip elements inside tables
        if element.find_parent('table'):
            continue
        
        # Get direct children (not all descendants)
        children = list(element.children)
        
        # Skip if no children or if it contains a table
        if not children or element.find('table', recursive=False):
            continue
        
        # Combine text from direct children, skipping nested tables
        combined_texts = []
        for child in children:
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    combined_texts.append(text)
            elif isinstance(child, Tag):
                # Skip if this child contains a table
                if child.find('table'):
                    continue
                child_text = normalize_text(child.get_text("", strip=True))
                if child_text:
                    combined_texts.append(child_text)
        
        if not combined_texts:
            continue
        
        combined_text = " ".join(combined_texts)
        spaceless_combined = combined_text.replace(" ", "")
        
        # Check length to avoid matching huge containers
        if len(spaceless_combined) > len(spaceless_target) * 10 and len(spaceless_combined) > 300:
            continue
        
        if spaceless_combined == spaceless_target:
            parent_matches.append((element, 'exact'))
            logger.info(f"Found exact match by combining children in '{element.name}' tag")
        elif spaceless_combined.startswith(spaceless_target):
            parent_matches.append((element, 'starts_with'))
            logger.debug(f"Found 'starts with' match by combining children in '{element.name}' tag")
    
    # Prioritize exact matches over starts_with
    exact_parent_matches = [m for m in parent_matches if m[1] == 'exact']
    if exact_parent_matches:
        return exact_parent_matches[0][0]
    
    starts_with_parent_matches = [m for m in parent_matches if m[1] == 'starts_with']
    if starts_with_parent_matches:
        logger.info(f"Found 'starts with' match by combining children")
        return starts_with_parent_matches[0][0]
    
    logger.warning(f"Could not find element for: '{target_text}'")
    return None

def find_next_table(element: Tag) -> Tag | None:
    """Find next data table after given element, skipping title wrapper tables."""
    if not element:
        return None
    
    current_table = element.find_next('table')
    
    while current_table:
        rows = current_table.find_all('tr')
        num_rows = len(rows)
        
        logger.debug(f"Examining table with {num_rows} rows")
        
        if num_rows < 3:
            logger.debug(f"Table too small ({num_rows} rows), looking for next")
            current_table = current_table.find_next('table')
            continue
        
        # Check if this is a title wrapper table
        if _is_title_wrapper_table(current_table):
            logger.info(f"Skipping title wrapper table, looking for next")
            current_table = current_table.find_next('table')
            continue
        
        # This appears to be a real data table
        logger.info(f"Found data table with {num_rows} rows")
        return current_table
    
    logger.warning("No suitable table found after element")
    return None


def _is_title_wrapper_table(table: Tag, title_element: Tag = None) -> bool:
    """
    Detect if a table is just a styling wrapper for a title.
    
    Args:
        table: The table to check
        title_element: Optional - if provided and is a <td>/<th> in this table,
                      this is DEFINITELY NOT a wrapper (it's a data table with title inside)
    
    Indicators of wrapper tables:
    - Few rows (typically 1-3)
    - Mostly empty cells
    - Only text content is a title (no numbers)
    - Large colspan usage
    
    Returns:
        True if this is a title wrapper, False if it's a data table
    """
    # CRITICAL: If title_element is a <td>/<th> in THIS table,
    # then this is NOT a wrapper - it's a data table with the title inside
    if title_element and title_element.name in ['td', 'th']:
        if title_element.find_parent('table') == table:
            logger.info("Title is a cell inside this table - NOT a wrapper")
            return False
    
    rows = table.find_all('tr')
    
    # Title wrappers usually have very few rows
    if len(rows) > 5:
        return False
    
    # Count cells with actual content
    cells_with_content = 0
    total_text = []
    #has_numbers = False
    
    for row in rows:
        cells = row.find_all(['td', 'th'])
        for cell in cells:
            text = cell.get_text(strip=True)
            if text:
                cells_with_content += 1
                total_text.append(text)
                # Check for numeric data
                if any(c.isdigit() for c in text):
                    has_numbers = True
    
    # If very few cells have content, likely a title wrapper
    if cells_with_content <= 2:
        combined_text = ' '.join(total_text)
        logger.debug(f"Title wrapper check: {cells_with_content} cells, text: '{combined_text[:50]}...'")
        return True
    
    
    return False



def score_title(title_text: str, keywords: dict) -> int:
    """
    Score title based on keyword matching.
    Simplified scoring: must-have keywords heavily weighted, penalties applied.
    """
    text = title_text.lower()
    score = 0
    
    # Must-have keywords
    must_have_matches = sum(1 for kw in keywords['must_have'] if kw in text)
    score += must_have_matches * 100
    
    # Nice-to-have keywords
    nice_to_have_matches = sum(1 for kw in keywords['nice_to_have'] if kw in text)
    score += nice_to_have_matches * 10
    
    # Apply penalties
    penalty_matches = sum(1 for kw in keywords['penalties'] if kw in text)
    score -= penalty_matches * 30
    
    # Penalize numbered items (likely notes or schedules)
    if re.match(r'^\s*\d{1,2}\s*[\.\)]', text):
        score -= 50
    
    # Slight penalty for very long titles
    score -= len(text) // 20
    
    return score

def find_financial_statement_table(semantic_tree: list, section_name: str,
                                      context: ScrapingContext,
                                      processed_tables: set, 
                                      used_title_texts: set) -> tuple[Tag, str, dict] | None:
    """
    Find financial statement table using semantic tree directly.
    No HTML searching needed!
    
    Returns:
        (table_bs4, title_text, metadata) or None
    """
    logger.info(f"\n--- Finding {section_name} (Direct Semantic Tree Method) ---")

    search_tree = semantic_tree
    
    # For 10-K, restrict search to after Item 8
    if context.form_type == '10-K':
        item8_index, _ = find_item8_node(semantic_tree)
        if item8_index != -1:
            search_tree = semantic_tree[item8_index:]
            logger.info(f"Restricted search to {len(search_tree)} nodes after Item 8")

    keywords = STATEMENT_KEYWORDS[section_name]
    matching_titles = find_statement_titles(search_tree, keywords)
    
    # Filter out already-used titles
    unprocessed_titles = [t for t in matching_titles if t['text'] not in used_title_texts]
    
    if not unprocessed_titles:
        logger.warning(f"No new titles found for {section_name}")
        return None

    # Score and prioritize
    prioritized_titles = sorted(
        unprocessed_titles,
        key=lambda t: score_title(t['text'], keywords),
        reverse=True
    )
    
    # Try each candidate
    for title_info in prioritized_titles:
        title_text = title_info['text']
        title_node = title_info['node']
        
        logger.info(f"Processing: '{title_text}'")
        
        # *** KEY CHANGE: Extract table directly from semantic tree ***
        result = extract_table_from_semantic_node(title_node)
        
        if not result:
            logger.warning(f"Could not find table for: '{title_text}'")
            continue
        
        table_bs4, metadata = result
        
        # Check if already processed
        if table_bs4 in processed_tables:
            logger.warning("Table already processed, trying next candidate")
            continue

        rows = len(table_bs4.find_all('tr'))
        
        # Skip very large tables in 10-K (likely ToC)
        if context.form_type == '10-K' and rows > 100:
            logger.warning(f"Skipping very large table ({rows} rows)")
            continue

        logger.info(f"✓ Found table for {section_name}")
        logger.info(f"  Source: {metadata['source']}")
        logger.info(f"  Extracted title: {metadata['is_extracted_title']}")
        logger.info(f"  Rows: {rows}")
        
        return table_bs4, title_text, metadata
    
    logger.error(f"Could not find table for {section_name}")
    return None



# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_data_from_table(table: Tag, title_text: str, metadata: dict,
                           section_name: str, filing_info: dict, 
                           context: ScrapingContext) -> list:
    """Extract data from table, skipping title row if it's inside the table."""
    if not table:
        logger.error(f"No table provided for '{section_name}'")
        return []
    
    logger.info(f"Extracting data from table (source: {metadata['source']})")
    
    metric_col = context.get_metric_column(section_name)
    is_snapshot = (section_name == 'Balance Sheet')
    
    # Build grid
    grid = build_grid_from_table(table)
    logger.debug(f"Table grid: {len(grid)} rows x {len(grid[0]) if grid else 0} cols")
    
    # Split header/body - pass title_row_index if we know it
    title_row_index = metadata.get('title_row_index')
    
    if title_row_index is not None:
        logger.info(f"Title is in row {title_row_index} of this table - will skip it")
    
    header_section, body_start_row = split_table_into_header_and_body(
        grid, metric_col, title_row_index
    )
    
    # Parse periods
    periods = []
    if header_section:
        logger.info(f"Parsing header for '{section_name}'")
        periods = parse_header(header_section, metric_col, context, is_snapshot)
    
    # Fallback strategies...
    if not periods and metadata.get('is_extracted_title'):
        logger.warning("Header parsing failed, looking for period info near table")
        periods = _parse_periods_near_table(table, context, is_snapshot)
    
    if not periods and title_row_index is not None:
        logger.warning("Header parsing failed, extracting from rows after title")
        # Try to parse from rows between title and body
        potential_header = grid[title_row_index + 1:body_start_row]
        if potential_header:
            periods = parse_header(potential_header, metric_col, context, is_snapshot)
    
    # Apply corrections
    if not is_snapshot and periods:
        periods = correct_fiscal_periods(periods)
    
    if not periods:
        logger.error(f"Could not detect periods for '{section_name}'")
        return []

    # Detect units and currency
    currency, units = auto_detect_units_and_currency(grid, table)
    logger.info(f"Detected Periods: {periods}")
    logger.info(f"Detected Units: {currency} in {units}")

    # Extract data - same as before
    data_records = []
    current_section_header = ""
    num_periods = len(periods)
    
    for row_idx, row in enumerate(grid[body_start_row:], start=body_start_row):
        if len(row) <= metric_col or not (metric := row[metric_col].strip()):
            continue

        values = extract_ordered_values_from_row(row, metric_col)
        
        if not values:
            current_section_header = metric
            continue
        
        # Align values with periods
        if len(values) != num_periods:
            logger.warning(f"Row {row_idx} ('{metric}'): "
                         f"Found {len(values)} values, expected {num_periods}")
            if len(values) < num_periods:
                values.extend([0.0] * (num_periods - len(values)))
            else:
                values = values[:num_periods]
        
        for period, value in zip(periods, values):
            data_records.append({
                **filing_info,
                'Financial Section': section_name,
                'Section': current_section_header,
                'Metric': metric,
                'Period': period,
                'Value': value,
                'Currency': currency,
                'Units': units,
            })
            
    logger.info(f"Extracted {len(data_records)} records from '{section_name}'")
    return data_records

def _parse_periods_near_table(table: Tag, context: ScrapingContext, 
                              is_snapshot: bool) -> list[str]:
    """
    Look for period information in elements immediately before the table.
    Useful when title was extracted by TableTitleSplitter.
    """
    # Look at previous siblings
    text_parts = []
    for sibling in table.find_previous_siblings(limit=5):
        if isinstance(sibling, Tag):
            text = sibling.get_text(" ", strip=True)
            if text:
                text_parts.append(text)
    
    if not text_parts:
        return []
    
    combined_text = " ".join(reversed(text_parts))  # Reverse to get chronological order
    return _parse_periods_from_text_blob(combined_text, context, is_snapshot)


def _find_text_between_elements(start_element: Tag, end_element: Tag) -> str:
    """Extracts all text between start and end HTML elements."""
    text_parts = []
    for sibling in start_element.find_next_siblings():
        if sibling is end_element:
            break
        if isinstance(sibling, NavigableString):
            if sibling.strip():
                text_parts.append(sibling.strip())
        elif isinstance(sibling, Tag):
            text_parts.append(sibling.get_text(" ", strip=True))
    return normalize_text(" ".join(text_parts))

def _parse_periods_from_text_blob(text: str, context: ScrapingContext, 
                                  is_snapshot: bool) -> list[str]:
    """
    Fallback: parse periods from text blob using regex.
    
    IMPROVED: Added constraints to avoid picking up spurious years.
    """
    if not text:
        return []
    
    # Only look at first 500 characters to avoid picking up unrelated years
    text = text[:500]
    
    date_pattern = re.compile(
        r"""
        (?:(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+)?
        \b(20\d{2})\b
        """,
        re.VERBOSE | re.IGNORECASE
    )
    
    periods = []
    current_year = datetime.now().year
    
    for match in date_pattern.finditer(text):
        full_match = match.group(0)
        year = int(match.group(2))
        
        # Sanity check: reject unrealistic years
        # Allow years from 10 years ago to 2 years in future
        if year < current_year - 10 or year > current_year + 2:
            logger.debug(f"Rejecting unrealistic year: {year}")
            continue
            
        period = parse_period(full_match, context, is_snapshot)
        if period and period not in periods:
            periods.append(period)
    
    # Limit to reasonable number of periods (max 10)
    if len(periods) > 10:
        logger.warning(f"Fallback parser found {len(periods)} periods - likely incorrect. Taking first 10.")
        periods = periods[:10]
    
    if not periods:
        logger.warning("Fallback parser found no valid periods")
    else:
        logger.info(f"Fallback parser found periods: {periods}")
    
    return periods



# ============================================================================
# MAIN SCRAPING ORCHESTRATION
# ============================================================================

def process_single_filing(filing_meta: dict, context: ScrapingContext) -> list:
    """Process one filing document using direct semantic tree access."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing {filing_meta['Filing Date']}")
    logger.info(f"URL: {filing_meta['URL']}") 
    logger.info(f"{'='*80}")

    # Get HTML and build semantic tree
    soup = get_soup(filing_meta['URL'])
    if not soup:
        return []

    html_content = str(soup)
    semantic_tree, _ = parse_document_with_semantic_tree(html_content, context.form_type)
    
    if not semantic_tree:
        logger.error("Failed to build semantic tree")
        return []

    filing_data = []
    processed_tables = set()
    used_title_texts = set()
    
    # Target sections
    target_sections = [
        'Income Statement',
        'Balance Sheet',
        'Cash Flow Statement',
        'Comprehensive Income Statement',
        'Stockholders Equity'
    ]

    try:
        for section_name in target_sections:
            logger.info(f"\n--- Searching for: {section_name} ---")

            # *** KEY CHANGE: Use semantic tree directly, no soup needed! ***
            result = find_financial_statement_table(
                semantic_tree, section_name, context,
                processed_tables, used_title_texts
            )
            
            if not result:
                logger.error(f"Could not find table for '{section_name}'")
                continue
            
            table_bs4, title_text, metadata = result
            processed_tables.add(table_bs4)
            used_title_texts.add(title_text)

            # Extract data - pass metadata for context
            records = extract_data_from_table(
                table_bs4, title_text, metadata,
                section_name, filing_meta, context
            )

            if records:
                filing_data.extend(records)
                logger.info(f"✓ Extracted {len(records)} records for '{section_name}'")
            else:
                logger.warning(f"No data extracted for '{section_name}'")
    
    except Exception as e:
        logger.error(f"Error processing filing: {e}")
        logger.debug(traceback.format_exc())
        save_semantic_tree_debug(semantic_tree, context.ticker, 
                                filing_meta['Filing Date'], reason='exception')
    
    if not filing_data:
        logger.warning("No data extracted - saving debug info")
        save_semantic_tree_debug(semantic_tree, context.ticker,
                                filing_meta['Filing Date'], reason='no_data')
    
    return filing_data

def run_scraping(ticker: str, form_type: str) -> list:
    """Main scraping orchestration."""
    all_data = []
    
    filings, fye = get_filing_urls(ticker, form_type=form_type)
    if not filings:
        return []

    # Generate fiscal calendar
    fye_month = 12  # Default
    if fye and len(fye) == 4:
        try:
            fye_month = int(fye[:2])
        except ValueError:
            logger.warning(f"Could not parse FYE month from '{fye}'")
    
    fiscal_calendar = generate_fiscal_calendar(fye_month)
    
    # Create scraping context
    context = ScrapingContext(
        fiscal_calendar=fiscal_calendar,
        ticker=ticker,
        form_type=form_type,
        metric_col_exceptions=METRIC_COLUMN_EXCEPTIONS
    )

    # Process filings (currently just the most recent one)
    for filing_meta in filings[:1]:
        try:
            records = process_single_filing(filing_meta, context)
            all_data.extend(records)
        except Exception as e:
            logger.error(f"Failed to process filing {filing_meta['Filing Date']}: {e}")
            logger.debug(traceback.format_exc())
            continue

    return all_data


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    logger.info("="*80)
    logger.info("SEC Financial Scraper - Simplified Edition")
    logger.info("="*80)

    try:
        ticker = input("\nEnter a stock ticker (e.g., AAPL, MSFT): ").strip().upper()
        if not ticker:
            logger.warning("No ticker entered. Exiting.")
            return

        form_type = ''
        while form_type not in ['10-Q', '10-K']:
            form_type = input("Enter report type ('10-Q' or '10-K'): ").strip().upper()
            if form_type not in ['10-Q', '10-K']:
                logger.warning("Invalid input. Please enter '10-Q' or '10-K'")

        # Run scraping
        scraped_data = run_scraping(ticker, form_type)

        if not scraped_data:
            logger.warning("\nScraping complete, but no data extracted")
            logger.info("Check 'scraper_debug.log' for detailed information")
            return

        # Save to CSV
        df = pd.DataFrame(scraped_data).drop_duplicates()
        output_filename = f"{ticker}_financials_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_filename, index=False)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"SUCCESS! Extracted {len(df)} unique data points")
        logger.info(f"Saved to: '{output_filename}'")
        logger.info(f"{'='*80}")
        
        # Display sample
        print("\n--- Sample of Extracted Data ---")
        print(df.head(10).to_string())

    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
    except Exception as e:
        logger.critical(f"Critical error: {e}", exc_info=True)


if __name__ == "__main__":
    main()