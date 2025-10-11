"""
SEC EDGAR API client for fetching filings and company information.
"""

import logging
import requests
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
from datetime import datetime

from .config import HEADERS

logger = logging.getLogger(__name__)


class SECClientError(Exception):
    """Base exception for SEC client errors."""
    pass


class SECClient:
    """Client for interacting with SEC EDGAR database."""
    
    def __init__(self, user_agent: str = None):
        """
        Initialize SEC client.
        
        Args:
            user_agent: Custom user agent string (overrides config.HEADERS)
        """
        self.headers = {'User-Agent': user_agent} if user_agent else HEADERS
    
    def get_cik_from_ticker(self, ticker: str) -> Optional[str]:
        """
        Convert stock ticker to CIK (Central Index Key).
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            CIK number as 10-digit string, or None if not found
        """
        try:
            url = 'https://www.sec.gov/files/company_tickers.json'
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            tickers_data = response.json()
            
            for company in tickers_data.values():
                if company['ticker'] == ticker.upper():
                    return str(company['cik_str']).zfill(10)
            
            logger.error(f"CIK not found for ticker: {ticker}")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching ticker-CIK mapping: {e}")
            return None
    
    def get_filings(
        self, 
        ticker: str, 
        form_type: str = '10-Q'
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        Fetch all filings and fiscal year-end for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            form_type: Type of form to fetch ('10-Q' or '10-K')
            
        Returns:
            Tuple of (filings_list, fiscal_year_end)
            filings_list: List of dicts with Filing Date, Report Period End Date, URL
            fiscal_year_end: Fiscal year-end as MMDD string (e.g., '1231')
        """
        cik = self.get_cik_from_ticker(ticker)
        if not cik:
            return [], None
        
        try:
            # Fetch main submissions index
            base_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            logger.info(f"Fetching submissions from {base_url}")
            
            response = requests.get(base_url, headers=self.headers)
            response.raise_for_status()
            submissions_data = response.json()
            
            # Extract fiscal year end
            fiscal_year_end = submissions_data.get('fiscalYearEnd')
            if fiscal_year_end:
                logger.info(f"Found Fiscal Year End for {ticker}: {fiscal_year_end}")
            else:
                logger.warning(f"Could not find Fiscal Year End for {ticker}")
            
            # Collect all filings (recent + archived)
            all_filings_data = []
            
            # Process recent filings
            recent_data = submissions_data.get('filings', {}).get('recent', {})
            if recent_data:
                keys = list(recent_data.keys())
                if keys:
                    num_recent = len(recent_data[keys[0]])
                    for i in range(num_recent):
                        filing_dict = {key: recent_data[key][i] for key in keys}
                        all_filings_data.append(filing_dict)
            
            # Process paginated archive files
            paginated_files = submissions_data.get('filings', {}).get('files', [])
            for file_meta in paginated_files:
                archive_url = f"https://data.sec.gov/submissions/{file_meta['name']}"
                logger.debug(f"Fetching archive: {archive_url}")
                
                archive_resp = requests.get(archive_url, headers=self.headers)
                archive_resp.raise_for_status()
                archive_data = archive_resp.json()
                
                keys = list(archive_data.keys())
                if keys:
                    num_filings = len(archive_data[keys[0]])
                    for i in range(num_filings):
                        filing_dict = {key: archive_data[key][i] for key in keys}
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
                        'URL': (
                            f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                            f"{accession_number.replace('-', '')}/"
                            f"{filing_data.get('primaryDocument', '')}"
                        )
                    })
                    unique_accession_numbers.add(accession_number)
            
            # Sort by filing date (most recent first)
            filings.sort(key=lambda x: x['Filing Date'], reverse=True)
            
            logger.info(f"Found {len(filings)} '{form_type}' filings for {ticker}")
            return filings, fiscal_year_end
            
        except Exception as e:
            logger.error(f"Failed to fetch filings: {e}")
            raise SECClientError(f"Failed to fetch filings for {ticker}") from e
    
    def fetch_filing_html(self, url: str) -> Optional[BeautifulSoup]:
        """
        Download and parse filing HTML.
        
        Args:
            url: URL of the filing document
            
        Returns:
            BeautifulSoup object or None if download failed
        """
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            return None


def generate_fiscal_calendar(fye_month: int) -> Dict[str, str]:
    """
    Generate fiscal month-to-quarter mapping based on fiscal year-end.
    
    Args:
        fye_month: Fiscal year-end month (1-12)
        
    Returns:
        Dict mapping month names to quarters (Q1-Q4)
    """
    if not 1 <= fye_month <= 12:
        fye_month = 12
        logger.warning(f"Invalid FYE month, defaulting to December")
    
    months = [
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december'
    ]
    
    # Reorder so fiscal year starts at the beginning
    fiscal_months = months[fye_month:] + months[:fye_month]
    
    calendar = {}
    calendar.update({m: 'Q1' for m in fiscal_months[0:3]})
    calendar.update({m: 'Q2' for m in fiscal_months[3:6]})
    calendar.update({m: 'Q3' for m in fiscal_months[6:9]})
    calendar.update({m: 'Q4' for m in fiscal_months[9:12]})
    
    logger.debug(f"Generated fiscal calendar for FYE month {fye_month}: {calendar}")
    return calendar