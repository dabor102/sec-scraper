import requests
import json

# The SEC requires a custom User-Agent for all API requests.
# Replace 'Your Name' and 'your.email@example.com' with your actual information.
# This helps the SEC identify the source of the traffic.
HEADERS = {'User-Agent': 'Your Name your.email@example.com'}

def get_cik_from_ticker(ticker: str) -> str:
    """
    Converts a stock ticker to its corresponding CIK number.

    The SEC provides a JSON file that maps all tickers to their CIKs.
    This function downloads, caches, and looks up the CIK for the given ticker.

    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL').

    Returns:
        The CIK number as a string, or None if not found.
    """
    url = 'https://www.sec.gov/files/company_tickers.json'
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()  # Raise an exception for bad status codes
        all_tickers = response.json()
        
        # The JSON is structured as { '0': {'cik_str': ..., 'ticker': ..., 'title': ...}, ... }
        # We iterate through the values to find the matching ticker.
        for company in all_tickers.values():
            if company['ticker'] == ticker.upper():
                # Pad CIK with leading zeros to make it 10 digits long
                return str(company['cik_str']).zfill(10)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching ticker-CIK mapping: {e}")
    except KeyError:
        print(f"Could not find CIK for ticker: {ticker}")
    
    return None

def get_sec_filings(ticker: str):
    """
    Fetches and displays recent 10-K and 10-Q filings for a given stock ticker.

    Args:
        ticker: The stock ticker symbol.
    """
    print(f"Attempting to find CIK for ticker: {ticker}...")
    cik = get_cik_from_ticker(ticker)

    if not cik:
        print(f"Could not retrieve CIK for '{ticker}'. Please check the ticker symbol.")
        return

    print(f"Found CIK for {ticker.upper()}: {cik}")
    
    # Construct the URL for the company's submission data
    url = f'https://data.sec.gov/submissions/CIK{cik}.json'
    
    try:
        print("Fetching recent filings from the SEC...")
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract the 'recent' filings data
        recent_filings = data['filings']['recent']
        
        print(f"\n--- Recent 10-K and 10-Q Filings for {ticker.upper()} ---")
        
        found_filings = False
        # Iterate through the filings and print the relevant ones
        for i in range(len(recent_filings['form'])):
            form_type = recent_filings['form'][i]
            
            if form_type in ['10-K', '10-Q']:
                found_filings = True
                filing_date = recent_filings['filingDate'][i]
                report_date = recent_filings['reportDate'][i]
                accession_number_no_dashes = recent_filings['accessionNumber'][i].replace('-', '')
                primary_document = recent_filings['primaryDocument'][i]
                
                # Construct the link to the filing's index page
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number_no_dashes}/{primary_document}"
                
                print(f"\n- Filing Type: {form_type}")
                print(f"  Filing Date: {filing_date}")
                print(f"  Report Period End Date: {report_date}")
                print(f"  Link: {filing_url}")

        if not found_filings:
            print("No recent 10-K or 10-Q filings were found in the latest data.")

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch data from SEC EDGAR. Error: {e}")
    except KeyError:
        print("Could not parse the filings data. The structure might have changed.")

def main():
    """
    Main function to run the script.
    """
    print("SEC Filing Fetcher")
    print("Enter a stock ticker to get its recent 10-K and 10-Q filings.")
    
    # Loop to allow multiple queries
    while True:
        ticker_input = input("\nEnter a stock ticker (or type 'exit' to quit): ").strip()
        if ticker_input.lower() == 'exit':
            break
        if not ticker_input:
            print("Please enter a ticker.")
            continue
            
        get_sec_filings(ticker_input)

if __name__ == "__main__":
    main()