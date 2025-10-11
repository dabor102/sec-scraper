"""
filing_style_analyzer.py - Analyze SEC filing styles using custom fetcher
"""

from bs4 import XMLParsedAsHTMLWarning
import warnings

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import requests
from sec_parser.processing_engine.core import Edgar10KParser, Edgar10QParser
from sec_parser.utils import DocumentStyleAnalyzer
from sec_parser.semantic_tree.tree_builder import TreeBuilder
from sec_parser.semantic_elements.title_element import TitleElement
from sec_parser.semantic_elements.table_element.table_element import TableElement
from sec_parser.semantic_elements.top_section_title import TopSectionTitle
import pandas as pd
from datetime import datetime


warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

import sys
from loguru import logger

# --- Loguru Configuration ---
# 1. Remove the default handler to prevent duplicate outputs.
logger.remove()

# 2. Add a new handler to output TRACE level logs to the console.
logger.add(
    sink="debug.log",
    level="TRACE",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)
# --- End of Loguru Configuration ---


# SEC requires custom User-Agent
HEADERS = {'User-Agent': 'Your Name your.email@example.com'}


def get_cik_from_ticker(ticker: str) -> str:
    """Convert ticker to CIK number."""
    url = 'https://www.sec.gov/files/company_tickers.json'
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        all_tickers = response.json()
        
        for company in all_tickers.values():
            if company['ticker'] == ticker.upper():
                return str(company['cik_str']).zfill(10)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching ticker-CIK mapping: {e}")
    except KeyError:
        print(f"Could not find CIK for ticker: {ticker}")
    
    return None


def get_latest_filing_url(ticker: str, form_type: str = '10-Q') -> tuple[str, dict]:
    """
    Get the URL of the latest filing for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        form_type: Type of form ('10-K' or '10-Q')
        
    Returns:
        Tuple of (filing_url, filing_metadata)
    """
    print(f"\nFetching latest {form_type} for {ticker.upper()}...")
    
    cik = get_cik_from_ticker(ticker)
    if not cik:
        print(f"Could not retrieve CIK for '{ticker}'")
        return None, None
    
    print(f"Found CIK: {cik}")
    
    url = f'https://data.sec.gov/submissions/CIK{cik}.json'
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        
        recent_filings = data['filings']['recent']
        
        # Find the first matching form type
        for i in range(len(recent_filings['form'])):
            if recent_filings['form'][i] == form_type:
                filing_date = recent_filings['filingDate'][i]
                report_date = recent_filings['reportDate'][i]
                accession_number = recent_filings['accessionNumber'][i]
                accession_number_no_dashes = accession_number.replace('-', '')
                primary_document = recent_filings['primaryDocument'][i]
                
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                    f"{accession_number_no_dashes}/{primary_document}"
                )
                
                metadata = {
                    'ticker': ticker.upper(),
                    'form_type': form_type,
                    'filing_date': filing_date,
                    'report_date': report_date,
                    'accession_number': accession_number,
                    'url': filing_url
                }
                
                print(f"\nFound {form_type} filing:")
                print(f"  Filing Date: {filing_date}")
                print(f"  Report Date: {report_date}")
                print(f"  URL: {filing_url}")
                
                return filing_url, metadata
        
        print(f"No {form_type} filings found")
        return None, None
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching filings: {e}")
        return None, None


def download_filing_html(url: str) -> str:
    """Download HTML content from SEC filing URL."""
    try:
        print(f"\nDownloading filing...")
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        html = response.text
        print(f"Downloaded {len(html):,} bytes")
        return html
    except requests.exceptions.RequestException as e:
        print(f"Error downloading filing: {e}")
        return None


def run_analysis_on_html(html: str, metadata: dict, form_type: str):
    """Core logic to analyze HTML content and print results."""
    print("\n" + "=" * 80)
    print(f"ANALYZING {metadata.get('ticker', 'URL')} {metadata.get('form_type', '')}")
    print(f"Filed: {metadata.get('filing_date', 'N/A')} | Period: {metadata.get('report_date', 'N/A')}")
    print("=" * 80)
    
    # Step 3: Analyze styles
    print("\n🔍 STYLE ANALYSIS")
    print("-" * 80)
    analyzer = DocumentStyleAnalyzer()
    results = analyzer.analyze(html)
    
    min_percentage = 0.5
    filtered_combinations = [
        stat for stat in results.style_combinations
        if stat.percentage >= min_percentage
    ]
    
    top_n = 20
    top_combinations = sorted(
        filtered_combinations, 
        key=lambda x: x.percentage, 
        reverse=True
    )[:top_n]

    if not top_combinations:
        print("No style combinations found meeting the criteria.")
    else:
        print(f"Displaying Top {len(top_combinations)} Style Combinations (>={min_percentage}% Occurrence)\n")
        for i, stat in enumerate(top_combinations, 1):
            print(f"Combination #{i}:")
            print(f"  Percentage: {stat.percentage:.2f}%")
            print(f"  Occurrences: {stat.occurrence_count}")
            print(f"  Avg text length: {stat.avg_text_length:.1f} chars")
            print(f"  Styles:")
            for key, value in sorted(stat.properties):
                print(f"    {key}: {value}")
            print("-" * 40)
    
    # Step 4: Parse with sec-parser
    print("\n\n📄 PARSING DOCUMENT")
    print("-" * 80)
    if form_type == '10-K':
        elements = Edgar10KParser().parse(html)
    else: # Default to 10-Q for URLs or other types
        elements = Edgar10QParser().parse(html)
    
    tree = TreeBuilder().build(elements)
    
    print(f"Parsed {len(elements)} elements")
    
    from collections import Counter
    element_types = Counter(type(e).__name__ for e in elements)
    print("\nElement type distribution:")
    for elem_type, count in element_types.most_common(10):
        print(f"  {elem_type}: {count}")
    
    titles = [e for e in elements if isinstance(e, TitleElement)]
    print(f"\n✓ Found {len(titles)} title elements")
    if titles:
        print("\nFirst 5 titles:")
        for i, title in enumerate(titles[:100], 1):
            print(f"  {i}. [{title.level}] {title.text[:80]}")
    
    # Step 5: Compare rare styles with detected titles
    print("\n\n🔬 STYLE vs TITLE DETECTION COMPARISON")
    print("-" * 80)
    rare_sizes = analyzer.get_rare_styles('font-size', max_percentage=10.0)
    rare_weights = analyzer.get_rare_styles('font-weight', max_percentage=10.0)
    
    print(f"\nRare font sizes (<5% of text):")
    for stat in rare_sizes[:5]:
        print(f"  {stat.value}: {stat.percentage:.2f}% "
              f"({stat.occurrence_count} occurrences, "
              f"avg {stat.avg_text_length:.0f} chars)")
    
    print(f"\nRare font weights (<5% of text):")
    for stat in rare_weights[:5]:
        print(f"  {stat.value}: {stat.percentage:.2f}% "
              f"({stat.occurrence_count} occurrences)")

    # Step 6: Save semantic tree
    if tree and metadata:
        print("\n💾 Saving semantic tree to CSV...")
        filename = save_semantic_tree_debug(
            semantic_tree=tree,
            ticker=metadata.get('ticker', 'URL_ANALYSIS'),
            form_type=metadata.get('form_type', 'CUSTOM'),
            filing_date=metadata.get('filing_date')
        )
        if filename:
            print(f"✓ Successfully saved to: {filename}")
        else:
            print("✗ Failed to save the semantic tree.")


def analyze_filing(ticker: str, form_type: str = '10-Q'):
    """Workflow for analyzing a filing by ticker."""
    filing_url, metadata = get_latest_filing_url(ticker, form_type)
    if not filing_url:
        return
    
    html = download_filing_html(filing_url)
    if not html:
        return
    
    run_analysis_on_html(html, metadata, form_type)


def analyze_url(url: str):
    """Workflow for analyzing a filing by direct URL."""
    html = download_filing_html(url)
    if not html:
        return

    # Create dummy metadata for the analysis function
    metadata = {
        'ticker': 'URL_INPUT',
        'form_type': 'custom',
        'filing_date': datetime.now().strftime('%Y-%m-%d'),
        'report_date': 'N/A',
        'url': url
    }
    
    # For URL analysis, we default to 10-Q parsing rules.
    run_analysis_on_html(html, metadata, '10-Q')


def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace while preserving content."""
    if not text:
        return ""
    return ' '.join(text.split())


def save_semantic_tree_debug(semantic_tree: list, ticker: str, form_type: str, 
                            filing_date: str = None, reason: str = None, 
                            section_name: str = None):
    """
    Saves the full, flattened semantic tree to CSV using recursion.
    """
    if not semantic_tree:
        print("Semantic tree is empty, nothing to save.")
        return None
    
    try:
        debug_records = []
        
        def traverse_and_add(node, level=0):
            if not hasattr(node, 'semantic_element'):
                return

            record = {
                'Level': level,
                'Element_Type': type(node.semantic_element).__name__,
                'Text': normalize_text(node.text),
                'Text_Length': len(node.text),
                'Children_Count': len(node.children),
            }
            debug_records.append(record)
            
            for child_node in node.children:
                traverse_and_add(child_node, level + 1)

        for top_level_node in semantic_tree:
            traverse_and_add(top_level_node)
        
        date_part = filing_date.replace('-', '') if filing_date else "NODATE"
        filename = f"{ticker.upper()}_{form_type}_{date_part}_semantic_tree.csv"
        
        df = pd.DataFrame(debug_records)
        df.to_csv(filename, index=False)
        
        return filename
        
    except Exception as e:
        print(f"Failed to save semantic tree debug file: {e}")
        return None

def main():
    """Interactive mode for analyzing filings."""
    print("=" * 80)
    print("SEC FILING STYLE ANALYZER")
    print("=" * 80)
    print("\nAnalyze styling patterns in SEC filings to identify title detection patterns.")
    
    while True:
        print("\n" + "-" * 80)
        user_input = input("\nEnter stock ticker, 'url' for a direct link, or 'exit' to quit: ").strip()
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        if not user_input:
            print("Please enter a ticker or command.")
            continue
            
        if user_input.lower() == 'url':
            pasted_url = input("Paste the full URL of the SEC filing to analyze: ").strip()
            if pasted_url:
                try:
                    analyze_url(pasted_url)
                except Exception as e:
                    print(f"\nAn unexpected error occurred during URL analysis: {e}")
            else:
                print("No URL provided.")
            
        else: # Assume it's a ticker
            ticker = user_input
            form_type = input("Enter form type (10-Q or 10-K) [default: 10-Q]: ").strip().upper()
            if not form_type:
                form_type = '10-Q'
            
            if form_type not in ['10-K', '10-Q']:
                print("Invalid form type. Using 10-Q.")
                form_type = '10-Q'
            
            try:
                analyze_filing(ticker, form_type)
            except (TypeError, AttributeError) as e:
                print(f"\nCould not analyze {ticker}. The filing may not be available or the format is unusual.")
                print(f"Error details: {e}")
                print("Please try another ticker.")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()