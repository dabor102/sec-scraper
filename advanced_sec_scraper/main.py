"""
Command-line interface for SEC financial scraper.
"""

import sys
import logging
import pandas as pd
from datetime import datetime

from .config import ScraperConfig
from .data_extractor import run_scraping


def setup_logging(verbose: bool = False):
    """
    Set up logging configuration.
    
    Args:
        verbose: If True, enable DEBUG level logging
    """
    logger = logging.getLogger("advanced_sec_scraper")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    if logger.handlers:
        return logger
    
    # File handler - always DEBUG
    file_handler = logging.FileHandler("scraper_debug.log", mode='w')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler - INFO or DEBUG based on verbose flag
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def main():
    """Main CLI entry point."""
    print("="*80)
    print("SEC Financial Scraper - Modular Edition")
    print("="*80)
    
    try:
        # Get user input
        ticker = input("\nEnter a stock ticker (e.g., AAPL, MSFT): ").strip().upper()
        if not ticker:
            print("No ticker entered. Exiting.")
            return
        
        form_type = ''
        while form_type not in ['10-Q', '10-K']:
            form_type = input("Enter report type ('10-Q' or '10-K'): ").strip().upper()
            if form_type not in ['10-Q', '10-K']:
                print("Invalid input. Please enter '10-Q' or '10-K'")
        
        # Ask about number of filings
        max_filings_input = input("\nHow many recent filings to process? (press Enter for all): ").strip()
        max_filings = None
        if max_filings_input:
            try:
                max_filings = int(max_filings_input)
                if max_filings <= 0:
                    print("Invalid number. Processing all filings.")
                    max_filings = None
            except ValueError:
                print("Invalid number. Processing all filings.")
                max_filings = None
        
        # Set up logging
        logger = setup_logging(verbose=False)
        
        # Create configuration
        config = ScraperConfig(max_filings=max_filings)
        
        # Run scraping
        logger.info(f"\nStarting scrape for {ticker} ({form_type})...")
        scraped_data = run_scraping(ticker, form_type, config)
        
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
        print("\nProcess interrupted by user")
    except Exception as e:
        logger = logging.getLogger("advanced_sec_scraper")
        logger.critical(f"Critical error: {e}", exc_info=True)
        print(f"\nAn error occurred. Check 'scraper_debug.log' for details.")


if __name__ == "__main__":
    main()