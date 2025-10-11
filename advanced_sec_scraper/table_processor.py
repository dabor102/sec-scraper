"""
Table processing utilities for extracting data from HTML tables.
"""

import logging
import re
from typing import List, Tuple, Optional
from bs4 import Tag

from .config import Patterns

logger = logging.getLogger(__name__)


# ============================================================================
# TEXT UTILITIES
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and invisible Unicode characters.
    
    This is the canonical version used throughout the scraper.
    """
    if not text:
        return ""
    
    # Remove invisible Unicode characters
    invisible_chars = [
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\ufeff',  # Byte order mark
    ]
    
    for char in invisible_chars:
        text = text.replace(char, '')
    
    # Standard whitespace normalization
    return ' '.join(text.split())


def is_empty_or_dash(cell_value: str) -> bool:
    """Check if a cell represents an empty/missing value."""
    if not cell_value or not cell_value.strip():
        return True
    stripped = cell_value.strip()
    dash_chars = {'-', '—', '–', '−', '―', '‐'}
    return stripped in dash_chars


# ============================================================================
# TABLE GRID CONSTRUCTION
# ============================================================================

class TableProcessor:
    """Processes HTML tables into structured grids."""
    
    @staticmethod
    def build_grid(table: Tag) -> List[List[str]]:
        """
        Build accurate 2D list from BeautifulSoup table tag.
        Handles colspan and rowspan correctly.
        
        Args:
            table: BeautifulSoup Tag representing a <table>
            
        Returns:
            2D list of cell values (strings)
        """
        grid = []
        
        for r, row_tag in enumerate(table.find_all('tr')):
            while len(grid) <= r:
                grid.append([])
            
            for cell_tag in row_tag.find_all(['th', 'td']):
                # Find next available column
                c = 0
                while c < len(grid[r]) and grid[r][c] is not None:
                    c += 1
                while c >= len(grid[r]):
                    grid[r].append(None)
                
                colspan = int(cell_tag.get('colspan', 1))
                rowspan = int(cell_tag.get('rowspan', 1))
                
                # Pre-process: remove hidden elements
                for hidden_tag in cell_tag.find_all(
                    style=lambda s: s and 'visibility:hidden' in s.lower()
                ):
                    hidden_tag.decompose()
                
                # Remove superscript tags (footnote markers)
                for sup_tag in cell_tag.find_all('sup'):
                    sup_tag.decompose()
                
                cell_text = normalize_text(cell_tag.get_text(separator=' ', strip=True))
                
                # Fill grid with cell value
                for i in range(rowspan):
                    for j in range(colspan):
                        while len(grid) <= r + i:
                            grid.append([])
                        while len(grid[r+i]) <= c + j:
                            grid[r+i].append(None)
                        grid[r+i][c+j] = cell_text if i == 0 and j == 0 else ""
        
        # Ensure rectangular grid
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
    
    @staticmethod
    def split_header_body(
        grid: List[List[str]], 
        metric_col: int,
        title_row_index: Optional[int] = None
    ) -> Tuple[List[List[str]], int]:
        """
        Split table grid into header and body sections.
        
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
        
        # Title indicators
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
        
        for row_idx, row in enumerate(grid):
            # Safety break
            if row_idx > 15:
                break
            
            # Skip title row if we know where it is
            if title_row_index is not None and row_idx == title_row_index:
                logger.debug(f"Skipping row {row_idx} (known title row)")
                continue
            
            if metric_col >= len(row):
                continue
            
            metric_text = normalize_text(row[metric_col])
            metric_text_lower = metric_text.lower()
            
            # Empty metric cell is part of header
            if not metric_text:
                continue
            
            # Check if this looks like a title row
            if any(indicator in metric_text_lower for indicator in title_indicators):
                logger.debug(f"Row {row_idx} appears to be title: '{metric_text}'")
                continue
            
            # Date/period keywords indicate header
            if any(keyword in metric_text_lower for keyword in header_keywords):
                continue
            
            # Unit declarations indicate header
            if Patterns.UNITS.match(metric_text):
                logger.debug(f"Row {row_idx} identified as header (units): '{metric_text}'")
                continue
            
            # Standalone years in metric column indicate header
            if Patterns.YEAR_ONLY.match(metric_text.strip()):
                logger.debug(f"Row {row_idx} identified as header (year): '{metric_text}'")
                continue
            
            # Check if metric text is a month name
            if any(month in metric_text_lower for month in month_names):
                data_cells = row[metric_col + 1:]
                has_years = any(Patterns.YEAR.search(cell) for cell in data_cells if cell.strip())
                
                if has_years:
                    logger.debug(f"Row {row_idx} identified as header (date with years): '{metric_text}'")
                    continue
                else:
                    if Patterns.YEAR.search(metric_text):
                        logger.debug(f"Row {row_idx} identified as header (complete date): '{metric_text}'")
                        continue
            
            # Check if data cells contain only years
            data_cells = row[metric_col + 1:]
            non_empty_data_cells = [cell.strip() for cell in data_cells if cell.strip()]
            
            if non_empty_data_cells:
                all_years = all(Patterns.YEAR_ONLY.match(cell) for cell in non_empty_data_cells)
                if all_years:
                    logger.debug(f"Row {row_idx} identified as header (row with only years): '{metric_text}'")
                    continue
            
            # This row has text but no header indicators - it's the body start
            logger.info(f"Body starts at row {row_idx}: '{metric_text}'")
            return grid[:row_idx], row_idx
        
        logger.warning("Could not determine header/body split. Defaulting to row 0.")
        return [], 0


# ============================================================================
# UNITS AND CURRENCY DETECTION
# ============================================================================

def extract_units_and_currency(text: str, check_symbols: bool = False) -> Tuple[str, str]:
    """
    Extract currency and unit information from text.
    
    Args:
        text: Text to analyze
        check_symbols: Whether to check for currency symbols ($, €, etc.)
        
    Returns:
        (currency, units) tuple
    """
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


def auto_detect_units_and_currency(
    grid: List[List[str]], 
    data_table: Tag
) -> Tuple[str, str]:
    """
    Automatically detect currency and units from table or surrounding text.
    
    Args:
        grid: Table grid
        data_table: BeautifulSoup table tag
        
    Returns:
        (currency, units) tuple
    """
    # Check table content
    table_text = normalize_text(" ".join(
        cell for row in grid[:15] for cell in row if cell
    ))
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

def parse_simple_value(text: str) -> Optional[float]:
    """
    Parse numeric value from text.
    
    Handles formats like:
    - "1,234.56"
    - "(1,234.56)" -> negative
    - "$1,234.56"
    """
    if not isinstance(text, str):
        return None
    
    text = text.strip()
    
    # Remove currency symbols and commas
    for symbol in ['$', '€', '£', '¥', '₹', ',']:
        text = text.replace(symbol, '')
    
    if is_empty_or_dash(text):
        return None
    
    # Check for negative (parentheses)
    is_negative = text.startswith('(') and text.endswith(')')
    if is_negative:
        text = text[1:-1]
    
    try:
        value = float(text)
        return -value if is_negative else value
    except ValueError:
        return None


def extract_ordered_values_from_row(
    row: List[str], 
    metric_col: int
) -> List[float]:
    """
    Extract numeric values in order from row, handling split negatives.
    
    Split negatives are when "(" is in one cell and "123)" is in another.
    
    Args:
        row: Table row
        metric_col: Index of metric column
        
    Returns:
        List of extracted values
    """
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