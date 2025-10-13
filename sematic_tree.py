import sec_parser as sp
import os

HTML_FILE_PATH = "/Users/dborn/Documents/GitHub/sec-scraper/test_filing.html"

# Utility function to make the example code a bit more compact
def print_first_n_lines(text: str, *, n: int):
    print("\n".join(text.split("\n")[:n]), "...", sep="\n")

# --- FIX: Read the HTML content from the file ---
try:
    with open(HTML_FILE_PATH, 'r', encoding='utf-8') as f:
        html_content = f.read()
except FileNotFoundError:
    print(f"Error: File not found at {HTML_FILE_PATH}")
    exit()

# Pass the content string instead of the file path
elements: list = sp.Edgar10QParser().parse(html_content)
# ------------------------------------------------

demo_output: str = sp.render(elements)
print_first_n_lines(demo_output, n=7)