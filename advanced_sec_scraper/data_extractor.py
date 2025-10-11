"""
Main orchestration for extracting financial data from SEC filings.
"""

import logging
import traceback
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import pandas as pd
from bs4 import BeautifulSoup

from .config import ScraperConfig, TARGET_SECTIONS, STATEMENT_KEYWORDS
from .sec_client import SECClient, generate_fiscal_calendar
from .statement_finder import StatementFinder
from .table_processor import TableProcessor, auto_detect_units_and_currency, extract_ordered_values_from_row
from .table_header_parser import TableHeaderParser, correct_fiscal_periods

logger = logging.getLogger(__name__)


# ============================================================================
# CONTEXT
# ============================================================================

@dataclass
class ScrapingContext:
    """Container for scraping state and configuration."""
    fiscal_calendar: Dict[str, str]
    ticker: str
    form_type: str
    config: ScraperConfig
    header_parser: TableHeaderParser = field(init=False)
    statement_finder: StatementFinder = field(init=False)
    table_processor: TableProcessor = field(init=False)
    
    def __post_init__(self):
        self.header_parser = TableHeaderParser(self.fiscal_calendar)
        self.statement_finder = StatementFinder(self.form_type)
        self.table_processor = TableProcessor()
    
    def get_metric_column(self, section_name: str, default: int = 0) -> int:
        """Get metric column for this ticker and section."""
        return self.config.get_metric_column(self.ticker, section_name, default)


# ============================================================================
# DEBUG UTILITIES
# ============================================================================

def save_semantic_tree_debug(
    semantic_tree: list,
    ticker: str,
    filing_date: str = None,
    reason: str = None
):
    """
    Save semantic tree to CSV for debugging.
    
    Args:
        semantic_tree: List of semantic tree nodes
        ticker: Stock ticker
        filing_date: Optional filing date
        reason: Reason for saving (e.g., 'no_item8', 'no_data')
    """
    if not semantic_tree:
        logger.warning("No semantic tree to save")
        return
    
    try:
        debug_records = []
        for idx, node in enumerate(semantic_tree):
            record = {
                'Index': idx,
                'Element_Type': type(node.semantic_element).__name__ if hasattr(node, 'semantic_element') else 'Unknown',
                'Text': node.text[:200] if hasattr(node, 'text') else '',
                'Text_Length': len(node.text) if hasattr(node, 'text') else 0,
                'Has_Children': len(node.children) if hasattr(node, 'children') else 0,
            }
            debug_records.append(record)
        
        # Include filing date in filename to avoid overwrites
        date_str = f"_{filing_date}" if filing_date else ""
        filename = f"{ticker}{date_str}_semantic_tree.csv"
        
        df = pd.DataFrame(debug_records)
        df.to_csv(filename, index=False)
        
        logger.info(f"Semantic tree saved for debugging: {filename}")
        logger.info(f"Reason: {reason}")
        logger.info(f"Total nodes saved: {len(debug_records)}")
        
        return filename
        
    except Exception as e:
        logger.error(f"Failed to save semantic tree debug file: {e}")


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_data_from_table(
    table,
    title_text: str,
    metadata: Dict,
    section_name: str,
    filing_info: Dict,
    context: ScrapingContext
) -> List[Dict]:
    """
    Extract data from table.
    
    Args:
        table: BeautifulSoup table tag
        title_text: Title of the statement
        metadata: Metadata about table location
        section_name: Name of financial statement
        filing_info: Filing metadata (date, URL, etc.)
        context: Scraping context
        
    Returns:
        List of data records
    """
    if not table:
        logger.error(f"No table provided for '{section_name}'")
        return []
    
    logger.info(f"Extracting data from table (source: {metadata['source']})")
    
    metric_col = context.get_metric_column(section_name)
    is_snapshot = (section_name == 'Balance Sheet')
    
    # Build grid
    grid = context.table_processor.build_grid(table)
    logger.debug(f"Table grid: {len(grid)} rows x {len(grid[0]) if grid else 0} cols")
    
    # Split header/body
    title_row_index = metadata.get('title_row_index')
    if title_row_index is not None:
        logger.info(f"Title is in row {title_row_index} - will skip it")
    
    header_section, body_start_row = context.table_processor.split_header_body(
        grid, metric_col, title_row_index
    )
    
    # Parse periods
    periods = []
    if header_section:
        logger.info(f"Parsing header for '{section_name}'")
        periods = context.header_parser.parse_header(header_section, metric_col, is_snapshot)
    
    # Apply corrections for non-snapshot statements
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


def process_single_filing(
    filing_meta: Dict,
    context: ScrapingContext,
    sec_client: SECClient
) -> List[Dict]:
    """
    Process one filing document.
    
    Args:
        filing_meta: Filing metadata (Filing Date, URL, etc.)
        context: Scraping context
        sec_client: SEC client for fetching HTML
        
    Returns:
        List of extracted data records
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing {filing_meta['Filing Date']}")
    logger.info(f"URL: {filing_meta['URL']}")
    logger.info(f"{'='*80}")
    
    # Get HTML - keep the full soup!
    soup = sec_client.fetch_filing_html(filing_meta['URL'])
    if not soup:
        return []
    
    html_content = str(soup)
    semantic_tree, _ = context.statement_finder.parse_document(html_content)
    
    if not semantic_tree:
        logger.error("Failed to build semantic tree")
        return []
    
    filing_data = []
    processed_tables = set()
    used_title_texts = set()
    
    try:
        for section_name in TARGET_SECTIONS:
            logger.info(f"\n--- Searching for: {section_name} ---")
            
            keywords = STATEMENT_KEYWORDS[section_name]
            
            # CRITICAL FIX: Pass the full soup so Strategy 3 can work!
            result = context.statement_finder.find_statement_table(
                semantic_tree, section_name, keywords,
                processed_tables, used_title_texts,
                full_soup=soup  # <-- This is the fix!
            )
            
            if not result:
                logger.error(f"Could not find table for '{section_name}'")
                continue
            
            table_bs4, title_text, metadata = result
            processed_tables.add(table_bs4)
            used_title_texts.add(title_text)
            
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
        save_semantic_tree_debug(
            semantic_tree, context.ticker,
            filing_meta['Filing Date'], reason='exception'
        )
    
    if not filing_data:
        logger.warning("No data extracted - saving debug info")
        save_semantic_tree_debug(
            semantic_tree, context.ticker,
            filing_meta['Filing Date'], reason='no_data'
        )
    
    return filing_data


# ============================================================================
# MAIN SCRAPING ORCHESTRATION
# ============================================================================

def run_scraping(
    ticker: str,
    form_type: str,
    config: Optional[ScraperConfig] = None,
    user_agent: str = None
) -> List[Dict]:
    """
    Main scraping orchestration.
    
    Args:
        ticker: Stock ticker symbol
        form_type: '10-Q' or '10-K'
        config: Optional scraper configuration
        user_agent: Optional custom user agent
        
    Returns:
        List of all extracted data records
    """
    if config is None:
        config = ScraperConfig()
    
    sec_client = SECClient(user_agent)
    
    # Get filings and fiscal year end
    filings, fye = sec_client.get_filings(ticker, form_type=form_type)
    if not filings:
        logger.error(f"No {form_type} filings found for {ticker}")
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
        config=config
    )
    
    # Process filings
    all_data = []
    max_filings = config.max_filings or len(filings)
    
    for filing_meta in filings[:max_filings]:
        try:
            records = process_single_filing(filing_meta, context, sec_client)
            all_data.extend(records)
        except Exception as e:
            logger.error(f"Failed to process filing {filing_meta['Filing Date']}: {e}")
            logger.debug(traceback.format_exc())
            continue
    
    return all_data