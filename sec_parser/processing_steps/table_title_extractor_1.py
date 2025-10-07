"""
TableTitleExtractor - Extracts titles from table structures.

This module is designed to identify and extract titles that are either embedded
within a table or for which a table serves as a stylistic wrapper. It's a common
practice in SEC filings to use tables to format titles, and this extractor can
distinguish between such cases and actual data tables.

The extractor operates as a step in the SEC parser's processing pipeline, transforming
`TableElement` objects into `TitleElement` objects when they are identified as titles.

Place this file in: sec_parser/processing_steps/table_title_extractor.py
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from sec_parser.processing_steps.abstract_classes.abstract_elementwise_processing_step import (
    AbstractElementwiseProcessingStep,
    ElementProcessingContext,
)
from sec_parser.semantic_elements.table_element.table_element import TableElement
from sec_parser.semantic_elements.title_element import TitleElement

# Assumes that header_parsing.py is in a location where it can be imported.
# This script contains your custom logic for identifying header rows.
from header_parsing import is_header_row

if TYPE_CHECKING:  # pragma: no cover
    from sec_parser.semantic_elements.abstract_semantic_element import (
        AbstractSemanticElement,
    )

class TableTitleExtractor(AbstractElementwiseProcessingStep):
    """
    The TableTitleExtractor is responsible for identifying tables that are used
    to display titles and converting them into TitleElement instances.
    """

    def _process_element(
        self,
        element: AbstractSemanticElement,
        context: ElementProcessingContext,
    ) -> AbstractSemanticElement:
        """
        Process a single semantic element. If the element is a table that functions
        as a title, it's converted to a TitleElement.
        """
        if not isinstance(element, TableElement):
            return element

        title = self._find_title_within_table(element)
        if title:
            # By creating a new TitleElement, we are effectively replacing the
            # TableElement in the final output.
            return TitleElement.create_from_element(
                element,
                # The text of the new title element is the title we extracted.
                text=title,
            )

        return element

    def _find_title_within_table(self, table_element: TableElement) -> str | None:
        """
        Analyzes a table to determine if it contains a title. This can be the case
        if the table is a "style wrapper" or if the title is "embedded" in the first row.
        """
        # We use the table's source HTML to analyze its internal structure.
        all_rows_in_table = table_element.html_tag._bs4.find_all('tr')
        
        non_empty_rows = []
        first_row_with_text = None

        for row in all_rows_in_table:
            row_text = self._get_text_from_row(row).strip()
            if row_text:
                non_empty_rows.append(row)
                if first_row_with_text is None:
                    first_row_with_text = {
                        "text": row_text,
                        "html": row,
                    }

        if not first_row_with_text:
            return None  # This table is empty

        # Scenario 1: The table is a "style wrapper" for the title.
        # This is true if the table contains only one row with text.
        if len(non_empty_rows) == 1:
            return first_row_with_text["text"]

        # Scenario 2: The title is "embedded" in the first non-empty row of a larger table.
        candidate_title = first_row_with_text["text"]
        candidate_row_html = first_row_with_text["html"]
        
        # Rule 1: The row should not be a header. We use your custom logic for this.
        if is_header_row(candidate_title):
            return None
        
        # Rule 2: The row should have title-like styling (bold or mostly uppercase).
        is_bold = self._is_row_bold(candidate_row_html)
        is_mostly_uppercase = self._is_mostly_uppercase(candidate_title)

        if is_bold or is_mostly_uppercase:
            return candidate_title

        return None

    def _get_text_from_row(self, row_html) -> str:
        """Extracts and concatenates text from all cells in a single table row."""
        cells = row_html.find_all(['td', 'th'])
        return ' '.join(cell.get_text(strip=True) for cell in cells)

    def _is_row_bold(self, row_html) -> bool:
        """
        Checks if the text within a row has bold styling. This is a strong
        indicator of a title.
        """
        if row_html.find(['b', 'strong']):
            return True
        
        # Also check for CSS font-weight styling
        style_pattern = re.compile(r"font-weight\s*:\s*(bold|700|800|900)")
        for tag in row_html.find_all(style=style_pattern):
            if tag.get_text(strip=True):
                return True
        return False
    
    def _is_mostly_uppercase(self, text: str, threshold: float = 0.8) -> bool:
        """
        Determines if a string is predominantly uppercase, which is another
        common characteristic of titles.
        """
        if not text:
            return False
        
        alpha_chars = [char for char in text if char.isalpha()]
        if not alpha_chars:
            return False
            
        uppercase_ratio = sum(1 for char in alpha_chars if char.isupper()) / len(alpha_chars)
        return uppercase_ratio >= threshold