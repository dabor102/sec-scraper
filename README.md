# Advanced SEC Financial Statement Scraper

An advanced Python-based web scraper designed to extract financial data from SEC 10-K and 10-Q filings. This tool leverages semantic parsing to intelligently locate and interpret financial statements, overcoming the challenges of inconsistent HTML structures in Edgar filings.

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Usage](#-usage)
- [Output Format](#-output-format)
- [Debugging](#-debugging)
- [Dependencies](#-dependencies)
- [Disclaimer](#-disclaimer)

---

## ✨ Key Features

* **Semantic Parsing**: Uses the `sec-parser` library to understand the document structure, identifying titles, sections, and tables based on their semantic role rather than just HTML tags.
* **Intelligent Table Detection**: Accurately finds the main financial statements (Income Statement, Balance Sheet, Cash Flow Statement, etc.) by scoring potential titles against a set of keywords and penalties.
* **Robust Header Parsing**: Deciphers complex, multi-row table headers to correctly identify financial periods (e.g., "Q3 2024", "FY 2023"). It can handle horizontal, vertical, split, and broadcast layouts.
* **Dynamic Fiscal Calendar**: Automatically generates a company's fiscal quarter mapping (Q1-Q4) based on its reported fiscal year-end, ensuring correct period labeling.
* **Automatic Data Normalization**: Cleans and parses numeric values, handling various accounting formats like parentheses for negative numbers `(1,234)` and removing currency symbols.
* **Smart Debugging**: If data extraction fails for a filing, the script automatically saves the document's semantic tree to a `.csv` file for easy troubleshooting.
* **Comprehensive Logging**: Provides clear console output for progress tracking and writes a detailed `scraper_debug.log` file for in-depth analysis.

---

## ⚙️ How It Works

The scraper follows a multi-step process to ensure accurate data extraction:

1.  **Fetch Filings**: It first queries the SEC EDGAR database to get a list of all `10-K` or `10-Q` filings for a given stock ticker. It also retrieves the company's fiscal year-end to build a fiscal calendar.
2.  **Semantic Analysis**: For a given filing, the HTML content is parsed into a **semantic tree**. Each element (e.g., a paragraph, a table) is classified by its function within the document.
3.  **Locate "Item 8"**: For `10-K` reports, the scraper specifically looks for **"Item 8. Financial Statements and Supplementary Data"** to narrow its search to the correct part of the document.
4.  **Identify Statement Titles**: It searches the semantic tree for titles that match keywords for the desired financial statements (e.g., "Consolidated Balance Sheets").
5.  **Find the Data Table**: Once a high-confidence title is found in the HTML, the scraper locates the first valid data table that follows it.
6.  **Parse and Extract**: The table is converted into a grid. The header rows are analyzed to determine the financial periods for each column. The script then iterates through the body rows, extracting the metric name and its corresponding values for each period.
7.  **Aggregate and Save**: The extracted data is compiled, cleaned, and saved to a final CSV file.

---

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    The script relies on a few key libraries. You can install them all with pip:
    ```bash
    pip install pandas requests beautifulsoup4 python-dateutil sec-parser
    ```

---

## 💻 Usage

The script runs interactively from the command line.

1.  **Run the script:**
    ```bash
    python advanced_sec_scraper.py
    ```

2.  **Enter the stock ticker** when prompted (e.g., `MSFT`, `GOOGL`).

3.  **Enter the report type** (`10-Q` or `10-K`).

The scraper will then process the most recent filing of the selected type. Upon completion, it will print a sample of the data and save the full results to a CSV file.

**Example Session:**
