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
        
        # Generate filename
        #timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        #date_part = f"_{filing_date.replace('-', '')}" if filing_date else ""
        #reason_part = f"_{reason}" if reason else ""
        #section_part = f"_{section_name.replace(' ', '_')}" if section_name else ""
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
# PERIOD PARSING - Consolidated
# ============================================================================

def parse_period(header_text: str, context: ScrapingContext, 
                is_snapshot: bool = False) -> str | None:
    """
    Unified period parser for both snapshot (balance sheet) and duration statements.
    Enhanced to handle month abbreviations and full date formats.
    
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
            return f"{found_quarter} {year}"
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
    
    Strategy:
    1. Find a row with date/period description but no years in data columns
    2. Look at next row(s) for actual year values
    3. Combine description with each year
    """
    logger.debug("Attempting split header format")
    
    # Patterns to identify description rows
    description_patterns = [
        re.compile(r'(?:fiscal\s+)?(?:years?|months?|quarters?)\s+ended', re.IGNORECASE),
        re.compile(r'(?:three|six|nine|twelve)\s+months?\s+ended', re.IGNORECASE),
        re.compile(r'as\s+of', re.IGNORECASE),
    ]
    
    year_pattern = re.compile(r'\b(19|20)\d{2}\b')
    
    for desc_row_idx in range(len(header_rows) - 1):  # -1 because we look ahead
        desc_row = header_rows[desc_row_idx]
        
        # Check all cells in this row for description text
        row_text = ' '.join(cell for cell in desc_row if cell.strip())
        row_text_lower = row_text.lower()
        
        # Does this row have a period description?
        has_description = any(pattern.search(row_text_lower) for pattern in description_patterns)
        
        if not has_description:
            continue
        
        # Check if data columns in this row are mostly empty or have no years
        data_cells = desc_row[metric_col + 1:]
        years_in_desc_row = [cell for cell in data_cells if cell.strip() and year_pattern.search(cell)]
        
        if years_in_desc_row:
            # Years are in the same row as description - not a split header
            continue
        
        logger.debug(f"Found description row {desc_row_idx}: '{row_text[:50]}...'")
        
        # Look at next few rows for years
        for year_row_idx in range(desc_row_idx + 1, min(desc_row_idx + 4, len(header_rows))):
            year_row = header_rows[year_row_idx]
            year_data_cells = year_row[metric_col + 1:]
            
            # Find cells with years
            year_cells = []
            for cell in year_data_cells:
                cell_stripped = cell.strip()
                if cell_stripped and year_pattern.search(cell_stripped):
                    year_cells.append(cell_stripped)
            
            if not year_cells:
                continue
            
            logger.debug(f"Found year row {year_row_idx} with years: {year_cells}")
            
            # Combine description with each year
            periods = []
            for year_cell in year_cells:
                combined_text = f"{row_text} {year_cell}"
                period = parse_period(combined_text, context, is_snapshot)
                if period and period not in periods:
                    periods.append(period)
            
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
                                     metric_col: int) -> tuple[list[list[str]], int]:
    """
    Splits table grid into header and body sections.
    
    Improved to better detect units declarations and avoid false positives.
    """
    if not grid:
        return [], 0

    header_keywords = {'year', 'ended', 'month', 'quarter', 'period', 'fiscal'}
    
    # Combined pattern for units declarations (both parenthetical and plain)
    units_pattern = re.compile(
    r'^\s*'  # Start of string, optional whitespace
    r'(?:'   # Non-capturing group for alternatives
        r'\([^)]*(?:millions|thousands|billions|shares|per\s+share|dollars)[^)]*\)|'  # Full parenthetical
        r'\(?\s*(?:amounts?\s+)?in\s+(?:thousands|millions|billions|shares|dollars|whole\s+dollars)'  # Plain format
    r')',
    re.IGNORECASE
)

    for row_idx, row in enumerate(grid):
        if row_idx > 15:  # Safety break
            break
        
        if metric_col >= len(row):
            continue

        metric_text = normalize_text(row[metric_col])
        metric_text_lower = metric_text.lower()

        # Empty metric cell is part of header
        if not metric_text:
            continue

        # Date/period keywords indicate header
        if any(keyword in metric_text_lower for keyword in header_keywords):
            continue
            
        # Unit declarations indicate header (both patterns)
        # Unit declarations indicate header
        if units_pattern.match(metric_text):
            logger.debug(f"Row {row_idx} identified as header (units): '{metric_text}'")
            continue
        
        # Check if row looks like standalone years (like "2023  2022  2021")
        # This happens when years are in the metric column
        year_only_pattern = re.compile(r'^(19|20)\d{2}$')
        if year_only_pattern.match(metric_text.strip()):
            logger.debug(f"Row {row_idx} identified as header (year): '{metric_text}'")
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
    """Finds an element by exact or 'starts with' match, excluding tables."""
    normalized_target = normalize_text(target_text)
    spaceless_target = normalized_target.replace(" ", "")
    
    if not spaceless_target:
        return None
    
    tags_to_search = ['p', 'div', 'h1', 'h2', 'h3', 'h4', 'font', 'b', 'strong']
    
    exact_matches = []
    starts_with_matches = []

    for element in soup.find_all(tags_to_search):
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
    
    if exact_matches:
        logger.info(f"Found exact match in '{exact_matches[0].name}' tag")
        return exact_matches[0]
    
    if starts_with_matches:
        logger.info(f"Found 'starts with' match in '{starts_with_matches[0].name}' tag")
        return starts_with_matches[0]
    
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


def _is_title_wrapper_table(table: Tag) -> bool:
    """
    Detect if a table is just a styling wrapper for a title.
    
    Indicators:
    - Few rows (typically 1-3)
    - Mostly empty cells
    - Only text content is a title (no numbers)
    - Large colspan usage
    """
    rows = table.find_all('tr')
    
    # Title wrappers usually have very few rows
    if len(rows) > 5:
        return False
    
    # Count cells with actual content
    cells_with_content = 0
    total_text = []
    has_numbers = False
    
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
    
    # Title wrappers don't have numeric data
    if not has_numbers and cells_with_content < 5:
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


def find_financial_statement_table(soup: BeautifulSoup, semantic_tree: list,
                                   section_name: str, context: ScrapingContext,
                                   processed_tables: set, 
                                   used_title_texts: set) -> tuple[Tag, Tag, str] | None:
    """
    Main function to find a financial statement table.
    """
    logger.info(f"\n--- Finding {section_name} ---")

    search_tree = semantic_tree
    
    # For 10-K, restrict search to after Item 8
    if context.form_type == '10-K':
        item8_index, _ = find_item8_node(semantic_tree)
        if item8_index != -1:
            search_tree = semantic_tree[item8_index:]
            logger.info(f"Restricted search to {len(search_tree)} nodes after Item 8")
        else:
            logger.warning("Could not find Item 8, searching entire document")

    keywords = STATEMENT_KEYWORDS[section_name]
    matching_titles = find_statement_titles(search_tree, keywords)
    
    # Filter out already-used titles
    unprocessed_titles = [t for t in matching_titles if t['text'] not in used_title_texts]
    
    if not unprocessed_titles:
        logger.warning(f"No new titles found for {section_name}")
        return None

    # Prioritize titles by score
    prioritized_titles = sorted(
        unprocessed_titles,
        key=lambda t: score_title(t['text'], keywords),
        reverse=True
    )

    if prioritized_titles:
        best = prioritized_titles[0]
        best_score = score_title(best['text'], keywords)
        logger.info(f"Best candidate: '{best['text']}' (Score: {best_score})")
    
    # Try each candidate
    for title_info in prioritized_titles:
        title_text = title_info['text']
        logger.info(f"Processing: '{title_text}'")
        
        title_element = find_text_in_html(soup, title_text)
        if not title_element:
            logger.warning(f"Could not locate title in HTML: '{title_text}'")
            continue
        
        potential_table = find_next_table(title_element)
        
        while potential_table:
            if potential_table in processed_tables:
                logger.warning("Table already processed, searching for next")
                potential_table = potential_table.find_next('table')
                continue

            rows = len(potential_table.find_all('tr'))
            
            # Skip very large tables in 10-K (likely ToC)
            if context.form_type == '10-K' and rows > 100:
                logger.warning(f"Skipping very large table ({rows} rows)")
                potential_table = potential_table.find_next('table')
                continue

            logger.info(f"Successfully found table for {section_name}")
            return potential_table, title_element, title_text
        
        logger.warning(f"No valid table after title: '{title_text}'")
    
    logger.error(f"Could not find table for {section_name}")
    return None


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_data_from_table(table: Tag, title_element: Tag, section_name: str,
                           filing_info: dict, context: ScrapingContext) -> list:
    """Extracts financial data from a table."""
    if not table:
        logger.error(f"No table provided for '{section_name}'")
        return []
    
    metric_col = context.get_metric_column(section_name)
    is_snapshot = (section_name == 'Balance Sheet')
    
    # Build grid
    grid = build_grid_from_table(table)
    logger.debug(f"Table grid preview for {section_name}:")
    for i, row in enumerate(grid[:5]):
        logger.debug(f"  Row {i}: {row[:5]}...")
    
    # Split header/body
    header_section, body_start_row = split_table_into_header_and_body(grid, metric_col)
    
    # Parse periods
    periods = []
    if header_section:
        logger.info(f"Parsing header for '{section_name}'")
        periods = parse_header(header_section, metric_col, context, is_snapshot)
    
    # Fallback: look for text between title and table
    if not periods and title_element:
        logger.warning("Header parsing failed, trying text between title and table")
        text_blob = _find_text_between_elements(title_element, table)
        if text_blob:
            periods = _parse_periods_from_text_blob(text_blob, context, is_snapshot)
    
    # Apply corrections for non-balance sheet statements
    if not is_snapshot and periods:
        periods = correct_fiscal_periods(periods)
    
    if not periods:
        logger.error(f"Could not detect periods for '{section_name}'")
        return []

    # Detect units and currency
    currency, units = auto_detect_units_and_currency(grid, table)
    logger.info(f"Detected Periods: {periods}")
    logger.info(f"Detected Units: {currency} in {units}")

    # Extract data
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
    """Process one filing document."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing {filing_meta['Filing Date']} "
               f"(Period: {filing_meta['Report Period End Date']})")
    logger.info(f"URL: {filing_meta['URL']}")
    logger.info(f"{'='*80}")

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
    
    # Track if Item 8 was found (for 10-K)
    item8_found = True
    if context.form_type == '10-K':
        item8_index, _ = find_item8_node(semantic_tree)
        if item8_index == -1:
            item8_found = False
            logger.warning("Item 8 not found - saving semantic tree for debugging")
            save_semantic_tree_debug(
                semantic_tree,
                context.ticker,
                filing_meta['Filing Date'],
                reason='no_item8'
            )

    # Process each target section
    target_sections = [
        
        'Income Statement',
        'Balance Sheet',
        'Cash Flow Statement',
        'Comprehensive Income Statement',
        'Stockholders Equity'
    ]

    sections_found = {}
    
    try:
        for section_name in target_sections:
            logger.info(f"\n--- Searching for: {section_name} ---")

            found_item = find_financial_statement_table(
                soup, semantic_tree, section_name, context,
                processed_tables, used_title_texts
            )
            
            if not found_item:
                logger.error(f"Could not find table for '{section_name}'")
                sections_found[section_name] = False
                continue
            
            sections_found[section_name] = True
            table, title_element, found_title_text = found_item
            processed_tables.add(table)
            used_title_texts.add(found_title_text)

            records = extract_data_from_table(
                table, title_element, section_name, filing_meta, context
            )

            if records:
                filing_data.extend(records)
                logger.info(f"Extracted {len(records)} records for '{section_name}'")
            else:
                logger.warning(f"No data extracted for '{section_name}'")
    
    except Exception as e:
        logger.error(f"Error processing filing: {e}")
        logger.debug(traceback.format_exc())
        
        # Save semantic tree for debugging
        save_semantic_tree_debug(
            semantic_tree, 
            context.ticker, 
            filing_meta['Filing Date'],
            reason='exception'
        )
        
        # Continue with partial data
        logger.warning("Continuing with partial data after error")
    
    # Check if we got any data
    if not filing_data:
        logger.warning("No data extracted from any section - saving semantic tree for debugging")
        
        # Determine most specific reason
        if not any(sections_found.values()):
            reason = 'no_sections_found'
        elif not item8_found:
            reason = 'no_item8_no_data'
        else:
            reason = 'no_data_extracted'
        
        save_semantic_tree_debug(
            semantic_tree,
            context.ticker,
            filing_meta['Filing Date'],
            reason=reason
        )
    
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