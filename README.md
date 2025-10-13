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

## 🔩 Function Workflow

The script's execution is divided into a series of modular functions, each with a specific responsibility. The high-level workflow follows a top-down approach, starting from broad orchestration and moving to detailed parsing tasks.

---

### 1. Orchestration Layer

This layer manages the overall process, from user input to final output.

* **`main()`**: The primary entry point.
    * Gathers user input for the **ticker** and **form type**.
    * Invokes `run_scraping()` to begin the core logic.
    * Receives the aggregated data, formats it into a pandas DataFrame, and saves it to a CSV file.

* **`run_scraping()`**: Manages the scraping task for a single ticker.
    * Calls `get_filing_urls()` to fetch a list of relevant filing documents from the SEC.
    * Calls `generate_fiscal_calendar()` to create a date-to-quarter mapping specific to the company's fiscal year.
    * Initializes a `ScrapingContext` object to pass configuration and state through the workflow.
    * Loops through the filings and calls `process_single_filing()` for each one.

---

### 2. Document Processing Layer

This layer is responsible for handling a single filing document.

* **`process_single_filing()`**: The main function for processing one filing URL.
    * `get_soup()`: Downloads the HTML content and parses it into a BeautifulSoup object for tag-based navigation.
    * `parse_document_with_semantic_tree()`: Uses the `sec-parser` library to create a high-level semantic tree of the document, identifying elements by their meaning (e.g., `TopSectionTitle`).
    * **Loops through target statements** ("Income Statement", "Balance Sheet", etc.) and attempts to find and extract data for each one using the functions below.

---

### 3. Table Identification Layer

This layer's goal is to accurately locate the correct HTML table for a given financial statement.

* **`find_financial_statement_table()`**: The core search engine for finding a specific statement.
    * `find_statement_titles()`: Searches the semantic tree for nodes that are likely titles of financial statements based on keywords.
    * `score_title()`: Ranks the candidate titles to find the best match and avoid notes or supplementary schedules.
    * `find_text_in_html()`: Locates the highest-scoring title element within the raw BeautifulSoup HTML.
    * `find_next_table()`: Traverses the HTML from the title's location to find the first subsequent `<table>` that appears to contain data.

---

### 4. Data Extraction Layer

Once a table is identified, this layer extracts the structured data from it.

* **`extract_data_from_table()`**: The primary function for parsing a `<table>` tag.
    * `build_grid_from_table()`: Converts the raw HTML table, including cells with `colspan` and `rowspan`, into an accurate 2D Python list (a grid).
    * `split_table_into_header_and_body()`: Analyzes the top of the grid to determine where the header section ends and the data rows begin.
    * `parse_header()`: A sophisticated parser that interprets the header rows to determine the **financial period** (e.g., "Q3 2024", "FY 2023") corresponding to each data column.
    * `auto_detect_units_and_currency()`: Scans the table and surrounding text to find the reporting units (e.g., "in millions") and currency.
    * `extract_ordered_values_from_row()`: Iterates through the data rows, parsing the metric name and the series of numerical values, correctly handling formats for negative numbers.

---

### 5. Utilities and Debugging

A collection of helper functions that support the entire process.

* **`save_semantic_tree_debug()`**: Called on failure, this function saves the semantic tree to a CSV file to help diagnose why a document could not be processed.
* **`parse_period()`, `correct_fiscal_periods()`, `normalize_text()`**: A suite of helper functions for cleaning text, interpreting dates, and standardizing period labels.



### 6. process flow
SEC Parser Processing Pipeline
The sec-parser library processes SEC EDGAR HTML documents by passing them through a pipeline of steps. Each step is responsible for a specific task, such as classifying elements, splitting them, or merging them. The pipeline is designed to be modular and customizable, allowing developers to add, remove, or replace steps as needed.


graph TD
    A[HTML Document] --> B{HtmlTagParser};
    B --> C[List of NotYetClassifiedElement];
    C --> D{IndividualSemanticElementExtractor};
    D --> E{ImageClassifier};
    E --> F{EmptyElementClassifier};
    F --> G{TableClassifier};
    G --> H{TableTitleSplitter};
    H --> I{TableOfContentsClassifier};
    I --> J{TopSectionManagerFor10Q};
    J --> K{IntroductorySectionElementClassifier};
    K --> L{TextClassifier};
    L --> M{HighlightedTextClassifier};
    M --> N{SupplementaryTextClassifier};
    N --> O{PageHeaderClassifier};
    O --> P{PageNumberClassifier};
    P --> Q{TitleClassifier};
    Q --> R{TextElementMerger};
    R --> S[List of Semantic Elements];
Processing Steps Explained

Here is a breakdown of what happens at each stage of the pipeline:

HtmlTagParser: The process begins with the HtmlTagParser class, which takes a raw HTML document as input. It uses the BeautifulSoup4 library to parse the HTML and create a list of HtmlTag objects. These objects are wrappers around BeautifulSoup's Tag objects and serve as the initial representation of the document's elements.

IndividualSemanticElementExtractor: This step is responsible for splitting a single HTML element that represents multiple semantic elements into multiple SemanticElement instances. This is crucial for maintaining the structural integrity of the document during parsing.

ImageClassifier: The ImageClassifier identifies elements that contain <img> tags and converts them into ImageElement instances. This step helps in distinguishing image content from other types of content in the document.

EmptyElementClassifier: This step scans for elements that do not contain any words and classifies them as EmptyElement instances. This is useful for filtering out irrelevant elements that are often used for spacing or formatting purposes.

TableClassifier: The TableClassifier identifies <table> elements and converts them into TableElement instances. It also applies a threshold to filter out tables that are too small to be considered meaningful data tables.

TableTitleSplitter: This step extracts title-like rows from within table elements. It is designed to handle cases where titles are embedded within tables, a common practice in SEC filings.

TableOfContentsClassifier: The TableOfContentsClassifier identifies tables that serve as a table of contents and converts them into TableOfContentsElement instances. This helps in distinguishing ToC tables from data tables.

TopSectionManagerFor10Q: This step is responsible for identifying the top-level sections of a 10-Q report, such as "Part I, Item 1. Financial Statements". It uses regular expressions to match section titles and then classifies the corresponding elements as TopSectionTitle instances.

IntroductorySectionElementClassifier: The IntroductorySectionElementClassifier is designed to classify elements that are located before the actual contents of the document. For a 10-Q report, this step will mark all elements that appear before the 'part1' section.

TextClassifier: This step classifies any remaining NotYetClassifiedElement instances that contain words as TextElement instances. This is a general-purpose step that ensures all remaining text content is properly classified.

HighlightedTextClassifier: The HighlightedTextClassifier identifies text elements with specific styling, such as bold or italic text, and converts them into HighlightedTextElement instances. This is an intermediate step in identifying title elements.

SupplementaryTextClassifier: This step is responsible for identifying and classifying supplementary text within a document, such as unit qualifiers, additional notes, and disclaimers. It looks for specific patterns, such as text enclosed in parentheses, to identify these elements.

PageHeaderClassifier: The PageHeaderClassifier identifies and classifies page headers within the document. It looks for common patterns in page headers, such as repeating text, to distinguish them from other content.

PageNumberClassifier: This step identifies and classifies page numbers within the document. It uses a combination of pattern matching and frequency analysis to identify page numbers accurately.

TitleClassifier: The TitleClassifier converts HighlightedTextElement instances into TitleElement instances. It uses the order of appearance of unique styles to determine the hierarchical level of each title.

TextElementMerger: This final step merges adjacent TextElement instances into a single element. This is intended to fix formatting artifacts where a single sentence is split across multiple HTML tags.

After passing through all these steps, the final output is a list of semantic elements that accurately represent the structure and content of the original SEC filing.



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
$ python advanced_sec_scraper.py
SEC Financial Scraper - Simplified Edition
Enter a stock ticker (e.g., AAPL, MSFT): AAPL
Enter report type ('10-Q' or '10-K'): 10-K

INFO: Fetching submissions from https://data.sec.gov/submissions/CIK0000320193.json ... INFO: SUCCESS! Extracted 543 unique data points INFO: Saved to: 'AAPL_financials_20251006.csv'
--- Sample of Extracted Data ---
Filing Date Report Period End Date                                                URL Financial Section                      Section                      Metric Period    Value Currency     Units
0  2024-10-28             2024-09-28  https://www.sec.gov/Archives/edgar/data/00003...  Income Statement  Net sales                      Net sales FY 2024  383285.0     USD  millions
1  2024-10-28             2024-09-28  https://www.sec.gov/Archives/edgar/data/00003...  Income Statement  Net sales                      Net sales FY 2023  394328.0     USD  millions


---

## 📊 Output Format

The output is a CSV file named `{TICKER}_financials_{YYYYMMDD}.csv`. The columns are:

| Column                 | Description                                                                 |
| ---------------------- | --------------------------------------------------------------------------- |
| `Filing Date`          | The date the report was filed with the SEC.                                 |
| `Report Period End Date` | The end date of the financial period covered by the report.               |
| `URL`                  | The direct URL to the SEC filing document.                                  |
| `Financial Section`    | The name of the financial statement (e.g., `Balance Sheet`).                |
| `Section`              | The subsection within the statement (e.g., `Current Assets`).               |
| `Metric`               | The specific line item (e.g., `Cash and cash equivalents`).                 |
| `Period`               | The financial period for the value (e.g., `Q3 2024`, `FY 2023`).            |
| `Value`                | The numeric value of the metric.                                            |
| `Currency`             | The currency of the value (e.g., `USD`).                                    |
| `Units`                | The units of the value (e.g., `millions`, `thousands`).                     |

---

## 🐛 Debugging

The script is designed to make debugging easier:

* **Log File**: A detailed log of operations, including warnings and errors, is saved to `scraper_debug.log`. This is the first place to look if you encounter issues.
* **Semantic Tree Dump**: If the scraper runs but fails to extract any data from a filing (e.g., it cannot find "Item 8" or any financial tables), it will automatically export the document's semantic tree structure to a file named `{TICKER}_semantic_tree.csv`. This file shows how the scraper interpreted the document's structure and is invaluable for understanding why a particular table was not found.

---

## 📦 Dependencies

* [pandas](https://pandas.pydata.org/)
* [requests](https://requests.readthedocs.io/en/latest/)
* [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
* [python-dateutil](https://dateutil.readthedocs.io/en/stable/)
* [sec-parser](https://pypi.org/project/sec-parser/)

---

## ⚠️ Disclaimer

This tool is intended for educational and research purposes only. The data is retrieved directly from the SEC EDGAR database. Always verify critical financial data with official sources. The user agent in the script (`'Your Name your.email@example.com'`) should be updated to reflect your own information as per the SEC's fair access guidelines.
