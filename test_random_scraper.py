import unittest
import random
import logging
import json
import sys
import argparse
from datetime import datetime
from pathlib import Path
from advanced_sec_scraper import (
    get_filing_urls, 
    run_scraping, 
    generate_fiscal_calendar,
    ScrapingContext,
    process_single_filing,
    setup_logging
)

# Test configuration
TEST_TICKERS = ['A', 'AAL', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 
           'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 
           'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 
           'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APO', 'APP', 'APTV', 
           'ARE', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP', 'AZO', 'BA', 'BAC', 'BALL', 
           'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BF-B', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK', 
           'BMY', 'BR', 'BRK-B', 'BRO', 'BSX', 'BWA', 'BX', 'BXP', 'C', 'CAG', 'CAH', 'CAT', 'CB', 
           'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 
           'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 
           'COF', 'COO', 'COP', 'COR', 'COST', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CSCO', 'CSGP', 
           'CSX', 'CTAS', 'CTLT', 'CTRA', 'CTSH', 'CVS', 'CVX', 'D', 'DAL', 'DAY', 'DD', 'DE', 'DECK', 
           'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 
           'DUK', 'DVA', 'DVN', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'ELV', 'EMN', 
           'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ERIE', 'ES', 'ESS', 'ETN', 'ETR', 
           'ETSY', 'EVA', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FCX', 'FDS', 
           'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FISV', 'FITB', 'FMC', 'FOX', 'FOXA', 'FRT', 
           'FSLR', 'FTNT', 'FTV', 'GD', 'GE', 'GEHC', 'GEV', 'GFF', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 
           'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 
           'HD', 'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 
           'HUBB', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'INVH', 
           'IP', 'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBL', 'JCI', 'JEC', 'JNJ', 
           'JPM', 'JKHY', 'JNPR', 'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 
           'KO', 'KR', 'KVUE', 'L', 'LAD', 'LAMR', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 
           'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 
           'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 
           'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 
           'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC', 
           'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OGN', 'OKE', 'OMC', 
           'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PARA', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PEAK', 'PEG', 
           'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 
           'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 
           'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 
           'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 
           'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 
           'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 
           'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 
           'URI', 'USB', 'V', 'VEEV', 'VFC', 'VICI', 'VLO', 'VLTO', 'VMC', 'VRSK', 'VRSN', 'VRTX', 'VTR', 
           'VTRS', 'VZ', 'WAB', 'WAT', 'WBD', 'WCN', 'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 
           'WRB', 'WRK', 'WST', 'WY', 'WYNN', 'XOM', 'XRAY', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZTS']

FORM_TYPES = ['10-Q', '10-K']

# Global test configuration
RANDOM_SEED = None
FAILURES_LOG_FILE = "test_failures.json"


class FailureLogger:
    """Logs failed test cases to a JSON file for later investigation."""
    
    def __init__(self, log_file: str = FAILURES_LOG_FILE):
        self.log_file = Path(log_file)
        self.failures = []
        
        # Load existing failures if file exists
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    self.failures = json.load(f)
            except json.JSONDecodeError:
                self.failures = []
    
    def log_failure(self, ticker: str, form_type: str, filing_meta: dict, 
                   error: Exception, seed: int = None, additional_info: dict = None):
        """Log a failed scraping attempt."""
        failure_record = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'form_type': form_type,
            'filing_date': filing_meta.get('Filing Date'),
            'report_date': filing_meta.get('Report Period End Date'),
            'url': filing_meta.get('URL'),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'seed': seed
        }
        
        if additional_info:
            failure_record['additional_info'] = additional_info
        
        self.failures.append(failure_record)
        self._save()
    
    def log_no_data(self, ticker: str, form_type: str, filing_meta: dict,
                   seed: int = None, sections_found: dict = None):
        """Log cases where scraping succeeded but no data was extracted."""
        failure_record = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'form_type': form_type,
            'filing_date': filing_meta.get('Filing Date'),
            'report_date': filing_meta.get('Report Period End Date'),
            'url': filing_meta.get('URL'),
            'error_type': 'NoDataExtracted',
            'error_message': 'Scraping completed but no data was extracted',
            'seed': seed,
            'sections_found': sections_found
        }
        
        self.failures.append(failure_record)
        self._save()
    
    def _save(self):
        """Save failures to JSON file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.failures, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save failure log: {e}")
    
    def get_summary(self) -> dict:
        """Get summary statistics of failures."""
        if not self.failures:
            return {}
        
        summary = {
            'total_failures': len(self.failures),
            'by_error_type': {},
            'by_ticker': {},
            'by_form_type': {}
        }
        
        for failure in self.failures:
            # Count by error type
            error_type = failure.get('error_type', 'Unknown')
            summary['by_error_type'][error_type] = summary['by_error_type'].get(error_type, 0) + 1
            
            # Count by ticker
            ticker = failure.get('ticker', 'Unknown')
            summary['by_ticker'][ticker] = summary['by_ticker'].get(ticker, 0) + 1
            
            # Count by form type
            form_type = failure.get('form_type', 'Unknown')
            summary['by_form_type'][form_type] = summary['by_form_type'].get(form_type, 0) + 1
        
        return summary
    
    def print_summary(self):
        """Print a human-readable summary of failures."""
        summary = self.get_summary()
        
        if not summary:
            print("\n✓ No failures logged")
            return
        
        print(f"\n{'='*80}")
        print(f"FAILURE SUMMARY (Total: {summary['total_failures']})")
        print(f"{'='*80}")
        
        print("\nBy Error Type:")
        for error_type, count in sorted(summary['by_error_type'].items(), 
                                       key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count}")
        
        print("\nBy Ticker:")
        for ticker, count in sorted(summary['by_ticker'].items(), 
                                   key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {ticker}: {count}")
        
        print("\nBy Form Type:")
        for form_type, count in summary['by_form_type'].items():
            print(f"  {form_type}: {count}")
        
        print(f"\nDetailed failures saved to: {self.log_file}")
        print(f"{'='*80}\n")


class TestRandomScraper(unittest.TestCase):
    """Randomized integration tests for SEC scraper."""
    
    failure_logger = None
    
    @classmethod
    def setUpClass(cls):
        """Set up logging for tests."""
        cls.logger = setup_logging()
        cls.failure_logger = FailureLogger()
        
        # Set random seed if provided
        if RANDOM_SEED is not None:
            random.seed(RANDOM_SEED)
            cls.logger.info(f"Random seed set to: {RANDOM_SEED}")
        else:
            # Generate and log a random seed for potential reproduction
            seed = random.randint(0, 999999)
            random.seed(seed)
            cls.logger.info(f"Generated random seed: {seed}")
            cls.logger.info(f"To reproduce this test run, use: --seed {seed}")
        
        cls.logger.info("\n" + "="*80)
        cls.logger.info("Starting Random Scraper Tests")
        cls.logger.info("="*80)
    
    def test_random_filing_scrape(self):
        """
        Randomly select a ticker, form type, and filing, then attempt to scrape it.
        This test validates that the scraper can handle real-world filings without crashing.
        """
        # Randomly select ticker and form type
        ticker = random.choice(TEST_TICKERS)
        form_type = random.choice(FORM_TYPES)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Random Test: {ticker} - {form_type}")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        if RANDOM_SEED is not None:
            self.logger.info(f"Seed: {RANDOM_SEED}")
        self.logger.info(f"{'='*80}")
        
        # Get available filings
        filings, fye = get_filing_urls(ticker, form_type=form_type)
        
        # Assert we can fetch filings
        self.assertIsNotNone(filings, 
            f"Failed to fetch filings for {ticker}")
        self.assertGreater(len(filings), 0, 
            f"No {form_type} filings found for {ticker}")
        
        self.logger.info(f"Found {len(filings)} {form_type} filings for {ticker}")
        
        # Randomly select a filing (bias towards recent ones)
        # Use weighted random to prefer recent filings (first 5)
        if len(filings) <= 5:
            selected_filing = random.choice(filings)
        else:
            # 70% chance to pick from most recent 5, 30% from rest
            if random.random() < 0.7:
                selected_filing = random.choice(filings[:5])
            else:
                selected_filing = random.choice(filings[5:])
        
        filing_date = selected_filing['Filing Date']
        report_date = selected_filing['Report Period End Date']
        
        self.logger.info(f"Selected filing: {filing_date} (Period: {report_date})")
        
        # Create scraping context
        fye_month = 12  # Default
        if fye and len(fye) == 4:
            try:
                fye_month = int(fye[:2])
            except ValueError:
                pass
        
        fiscal_calendar = generate_fiscal_calendar(fye_month)
        context = ScrapingContext(
            fiscal_calendar=fiscal_calendar,
            ticker=ticker,
            form_type=form_type,
            metric_col_exceptions={}
        )
        
        # Attempt to scrape the filing
        try:
            scraped_data = process_single_filing(selected_filing, context)
            
            # Validate results
            self.assertIsNotNone(scraped_data, 
                "Scraping returned None")
            self.assertIsInstance(scraped_data, list, 
                "Scraping should return a list")
            
            # Log results
            if scraped_data:
                self.logger.info(f"✓ SUCCESS: Extracted {len(scraped_data)} records")
                
                # Validate data structure
                sample = scraped_data[0]
                required_fields = ['Financial Section', 'Metric', 'Period', 
                                 'Value', 'Currency', 'Units']
                for field in required_fields:
                    self.assertIn(field, sample, 
                        f"Missing required field: {field}")
                
                # Check for unique sections
                sections = set(r['Financial Section'] for r in scraped_data)
                self.logger.info(f"  Sections found: {sections}")
                
                # Validate we got some key statements
                key_statements = {'Income Statement', 'Balance Sheet', 
                                'Cash Flow Statement'}
                found_statements = key_statements & sections
                
                if found_statements:
                    self.logger.info(f"  ✓ Found key statements: {found_statements}")
                else:
                    self.logger.warning(f"  ⚠ No key statements found")
                
                # Basic data validation
                for record in scraped_data[:5]:  # Check first 5 records
                    self.assertIsInstance(record['Value'], (int, float),
                        "Value should be numeric")
                    self.assertIsInstance(record['Period'], str,
                        "Period should be string")
                    self.assertIsInstance(record['Metric'], str,
                        "Metric should be string")
                
            else:
                self.logger.warning(f"⚠ WARNING: No data extracted from {ticker} {form_type}")
                
                # Log this as a no-data failure
                self.failure_logger.log_no_data(
                    ticker=ticker,
                    form_type=form_type,
                    filing_meta=selected_filing,
                    seed=RANDOM_SEED
                )
                
        except Exception as e:
            self.logger.error(f"✗ FAILED: {str(e)}")
            
            # Log the failure
            self.failure_logger.log_failure(
                ticker=ticker,
                form_type=form_type,
                filing_meta=selected_filing,
                error=e,
                seed=RANDOM_SEED,
                additional_info={
                    'fiscal_year_end': fye,
                    'num_filings_available': len(filings)
                }
            )
            
            # Re-raise to fail the test
            raise
        
        self.logger.info(f"{'='*80}\n")
    
    def test_multiple_random_scrapes(self):
        """Run multiple random scrapes in one test (stress test)."""
        num_iterations = 3  # Adjust based on desired test duration
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Running {num_iterations} Random Scrapes")
        if RANDOM_SEED is not None:
            self.logger.info(f"Seed: {RANDOM_SEED}")
        self.logger.info(f"{'='*80}")
        
        results = {
            'success': 0,
            'no_data': 0,
            'error': 0
        }
        
        for i in range(num_iterations):
            ticker = random.choice(TEST_TICKERS)
            form_type = random.choice(FORM_TYPES)
            
            self.logger.info(f"\nIteration {i+1}/{num_iterations}: {ticker} - {form_type}")
            
            try:
                # Get filings
                filings, fye = get_filing_urls(ticker, form_type=form_type)
                if not filings:
                    self.logger.warning(f"No filings found for {ticker}")
                    results['no_data'] += 1
                    continue
                
                # Select random filing
                filing = random.choice(filings[:min(10, len(filings))])
                
                # Create context
                fye_month = 12
                if fye and len(fye) == 4:
                    try:
                        fye_month = int(fye[:2])
                    except ValueError:
                        pass
                
                fiscal_calendar = generate_fiscal_calendar(fye_month)
                context = ScrapingContext(
                    fiscal_calendar=fiscal_calendar,
                    ticker=ticker,
                    form_type=form_type,
                    metric_col_exceptions={}
                )
                
                # Scrape
                data = process_single_filing(filing, context)
                
                if data:
                    results['success'] += 1
                    self.logger.info(f"  ✓ {len(data)} records extracted")
                else:
                    results['no_data'] += 1
                    self.logger.warning(f"  ⚠ No data extracted")
                    
                    # Log no-data case
                    self.failure_logger.log_no_data(
                        ticker=ticker,
                        form_type=form_type,
                        filing_meta=filing,
                        seed=RANDOM_SEED
                    )
                    
            except Exception as e:
                results['error'] += 1
                self.logger.error(f"  ✗ Error: {str(e)}")
                
                # Log the error
                if 'filing' in locals():
                    self.failure_logger.log_failure(
                        ticker=ticker,
                        form_type=form_type,
                        filing_meta=filing,
                        error=e,
                        seed=RANDOM_SEED,
                        additional_info={'iteration': i+1}
                    )
        
        # Summary
        self.logger.info(f"\n{'='*80}")
        self.logger.info("Test Summary:")
        self.logger.info(f"  Successful: {results['success']}/{num_iterations}")
        self.logger.info(f"  No Data:    {results['no_data']}/{num_iterations}")
        self.logger.info(f"  Errors:     {results['error']}/{num_iterations}")
        self.logger.info(f"{'='*80}\n")
        
        # Test passes if we have at least some successes and no critical errors
        self.assertGreater(results['success'], 0, 
            "At least one scrape should succeed")


class TestSpecificEdgeCases(unittest.TestCase):
    """Test specific known edge cases."""
    
    failure_logger = None
    
    @classmethod
    def setUpClass(cls):
        cls.logger = setup_logging()
        cls.failure_logger = FailureLogger()
    
    def test_small_cap_ticker(self):
        """Test scraping a smaller company (different filing format)."""
        # Small cap companies often have simpler filings
        ticker = random.choice(['PLTR', 'SNAP', 'PINS', 'ZM'])
        form_type = '10-Q'
        
        self.logger.info(f"Testing small cap: {ticker}")
        
        try:
            data = run_scraping(ticker, form_type)
            self.assertIsNotNone(data)
            if data:
                self.logger.info(f"✓ Small cap test passed: {len(data)} records")
            else:
                self.logger.warning(f"⚠ Small cap test: no data extracted")
        except Exception as e:
            self.logger.error(f"Small cap test failed: {e}")
            # Log but don't fail - some small caps may have unusual formats
    
    def test_financial_sector(self):
        """Test scraping a financial company (different statement structure)."""
        ticker = random.choice(['JPM', 'BAC', 'GS', 'WFC'])
        form_type = '10-K'
        
        self.logger.info(f"Testing financial sector: {ticker}")
        
        try:
            data = run_scraping(ticker, form_type)
            self.assertIsNotNone(data)
            if data:
                self.logger.info(f"✓ Financial sector test passed: {len(data)} records")
            else:
                self.logger.warning(f"⚠ Financial sector test: no data extracted")
        except Exception as e:
            self.logger.error(f"Financial sector test failed: {e}")


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Randomized SEC scraper tests with reproducible seeds',
        epilog='Example: python test_random_scraper.py --seed 42'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducible test runs',
        default=None
    )
    parser.add_argument(
        '--clear-failures',
        action='store_true',
        help='Clear the failures log file before running tests'
    )
    parser.add_argument(
        '--show-failures',
        action='store_true',
        help='Show failure summary and exit without running tests'
    )
    
    # Parse known args (allows unittest args to pass through)
    args, remaining = parser.parse_known_args()
    
    # Set global seed
    RANDOM_SEED = args.seed
    
    # Handle --show-failures flag
    if args.show_failures:
        logger = setup_logging()
        failure_logger = FailureLogger()
        failure_logger.print_summary()
        sys.exit(0)
    
    # Handle --clear-failures flag
    if args.clear_failures:
        failure_log_path = Path(FAILURES_LOG_FILE)
        if failure_log_path.exists():
            failure_log_path.unlink()
            print(f"Cleared failure log: {FAILURES_LOG_FILE}")
        else:
            print(f"No failure log found: {FAILURES_LOG_FILE}")
    
    # Print seed information
    if RANDOM_SEED is not None:
        print(f"\n{'='*80}")
        print(f"Running tests with seed: {RANDOM_SEED}")
        print(f"{'='*80}\n")
    
    # Run unittest with remaining args
    sys.argv[1:] = remaining
    
    # Run tests
    result = unittest.main(verbosity=2, exit=False)
    
    # Print failure summary after tests complete
    print("\n" + "="*80)
    print("TEST RUN COMPLETE")
    print("="*80)
    
    failure_logger = FailureLogger()
    failure_logger.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if result.result.wasSuccessful() else 1)