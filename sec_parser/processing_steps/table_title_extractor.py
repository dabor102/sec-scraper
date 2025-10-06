"""
TableTitleExtractor - Extracts titles from table structures.

This module handles cases where titles are wrapped in table elements for styling,
such as in Intel's SEC filings. It uses structural analysis to distinguish between:
1. Title wrapper tables (should be extracted)
2. Data tables with headers (should be kept intact)

Place this in: sec_parser/processing_steps/table_title_extractor.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import bs4

from sec_parser.processing_engine.html_tag import HtmlTag
from sec_parser.processing_steps.abstract_classes.abstract_elementwise_processing_step import (
    AbstractElementwiseProcessingStep,
    ElementProcessingContext,
)
from sec_parser.semantic_elements.composite_semantic_element import (
    CompositeSemanticElement,
)
from sec_parser.semantic_elements.semantic_elements import NotYetClassifiedElement

from sec_parser.semantic_elements.title_element import TitleElement

if TYPE_CHECKING:  # pragma: no cover
    from sec_parser.semantic_elements.abstract_semantic_element import (
        AbstractSemanticElement,
    )


class TableTitleExtractor(AbstractElementwiseProcessingStep):
    """
    Extracts title-like rows from tables based on structural analysis.

    Detects tables used purely as styling wrappers for titles, while preserving
    actual data tables with their column headers intact.

    Key Features:
    - Identifies title wrapper tables (1-2 rows with styling, no data)
    - Distinguishes data tables from title wrappers
    - Preserves column headers in data tables
    - No hardcoded keywords or font-size thresholds
    - Structure-based detection (colspan, background-color, cell patterns)

    Examples:
        # Title wrapper (extracted):
        <table>
            <tr><td colspan="3" bgcolor="#004a86">Report of...</td></tr>
        </table>

        # Data table (kept intact):
        <table>
            <tr><td colspan="15">Gains (Losses) on Derivatives...</td></tr>
            <tr><td>Years</td><td>2024</td><td>2023</td></tr>
            <tr><td>Revenue</td><td>$100M</td><td>$90M</td></tr>
        </table>
    """

    def __init__(
        self,
        *,
        types_to_process: set[type[AbstractSemanticElement]] | None = None,
        types_to_exclude: set[type[AbstractSemanticElement]] | None = None,
    ) -> None:
        """
        Initialize the TableTitleExtractor.

        Args:
            types_to_process: Set of element types to process (default: {NotYetClassifiedElement})
            types_to_exclude: Set of element types to exclude from processing
        """
        super().__init__(
            types_to_process=types_to_process or {NotYetClassifiedElement},
            types_to_exclude=types_to_exclude,
        )

    def _process_element(
        self,
        element: AbstractSemanticElement,
        _: ElementProcessingContext,
    ) -> AbstractSemanticElement:
        """
        Process element to extract title rows from tables if present.

        Args:
            element: The semantic element to process
            _: Processing context (unused)

        Returns:
            Either the original element, or a CompositeSemanticElement containing
            extracted titles and remaining table content
        """
        if not element.html_tag.contains_tag("table", include_self=True):
            return element

        title_rows, remaining_table = self._extract_title_rows(element.html_tag)

        if not title_rows:
            return element

        # Create new elements for the extracted titles and remaining table
        new_elements = []

        for title_row in title_rows:
            title_element = TitleElement(
                title_row,
                processing_log=element.processing_log.copy(),
                log_origin=self.__class__.__name__,
            )
            title_element.processing_log.add_item(
                message="Extracted as title row from table",
                log_origin=self.__class__.__name__,
            )
            new_elements.append(title_element)

        # Add remaining table if it has content
        if remaining_table and self._has_meaningful_content(remaining_table):
            table_element = NotYetClassifiedElement(
                remaining_table,
                processing_log=element.processing_log.copy(),
                log_origin=self.__class__.__name__,
            )
            new_elements.append(table_element)

        # Return composite element containing both title and table
        return CompositeSemanticElement.create_from_element(
            element,
            log_origin=self.__class__.__name__,
            inner_elements=new_elements,
        )
    
    
    def _extract_title_rows(
        self,
        html_tag: HtmlTag
    ) -> tuple[list[HtmlTag], HtmlTag | None]:
        """
        Extract rows that look like titles from a table.

        Strategy: Identify tables/rows used purely for title styling.
        Only extract from title wrapper tables, NOT from data tables.

        Args:
            html_tag: The HTML tag to analyze

        Returns:
            tuple: (list of title HtmlTags, remaining table HtmlTag or None)
        """
        bs4_tag = html_tag._bs4  # Access underlying bs4 tag

        if bs4_tag.name != "table":
            table = bs4_tag.find("table")
            if not table:
                return [], None
        else:
            table = bs4_tag

        # Get all rows with text content
        all_rows = table.find_all("tr")
        content_rows = [row for row in all_rows if row.get_text(strip=True)]

        # CRITICAL CHECK: Is this a data table?
        # If yes, don't extract headers - they belong with the data
        #if self._is_data_table(content_rows):
       #     return [], None

        # Case 1: Table has only 1-2 rows with text (likely just a title wrapper)
        if len(content_rows) <= 2:
            # Check if it's really just a title (short text, has styling)
            if self._is_single_title_table(content_rows):
                # Extract all content rows as titles
                title_rows = []
                for row in content_rows:
                    content = self._extract_title_content(row)
                    if content:
                        title_rows.append(HtmlTag(content))
                return title_rows, None

        # If the conditions above are not met, this is not a title table.
        # Return an empty list to signify no titles were extracted.
        return [], None

    def _is_single_title_table(self, content_rows: list[bs4.Tag]) -> bool:
        """
        Check if table with 1-2 rows is just a title wrapper.

        Criteria:
        - Short combined text (< 200 chars)
        - Has visual styling (background-color, large colspan)
        - Not a data table (no multiple columns with data)

        Args:
            content_rows: List of table rows with text content

        Returns:
            True if this is a title wrapper, False otherwise
        """
        if not content_rows:
            return False

        # Get combined text from all content rows
        combined_text = " ".join(row.get_text(strip=True) for row in content_rows)

        # Short text suggests title, not data
        if len(combined_text) > 200:
            return False

        # Check for styling indicators
        has_styling = False
        for row in content_rows:
            cells = row.find_all(["td", "th"])

            # Large colspan suggests title (spanning across layout)
            has_large_colspan = any(
                cell.get("colspan") and int(cell.get("colspan", 1)) >= 3
                for cell in cells
            )

            # Background color suggests styled title
            has_background = any(
                "background-color" in cell.get("style", "")
                for cell in cells
            )

            # Font styling suggests title
            has_font_styling = any(
                "font-size" in str(cell.get("style", "")) or
                "font-weight" in str(cell.get("style", ""))
                for cell in cells
            )

            if has_large_colspan or has_background or has_font_styling:
                has_styling = True
                break

        # Check it's not a data table (avoid false positives)
        # Data tables typically have multiple cells per row with different content
        is_not_data_table = True
        for row in content_rows:
            cells_with_text = [c for c in row.find_all(["td", "th"]) if c.get_text(strip=True)]
            # If multiple cells with substantial different content, likely data
            if len(cells_with_text) >= 3:
                texts = [c.get_text(strip=True) for c in cells_with_text]
                # Check if they're not just spacing cells
                substantial_texts = [t for t in texts if len(t) > 5]
                if len(substantial_texts) >= 2:
                    is_not_data_table = False
                    break

        return has_styling and is_not_data_table

    def _extract_title_content(self, row: bs4.Tag) -> bs4.Tag | None:
        """
        Extract the actual title content from a row.

        Returns the innermost element with the actual styled text,
        unwrapping layout cells.

        Args:
            row: The table row to extract content from

        Returns:
            The bs4.Tag containing the title content, or None if not found
        """
        # Get the main cell with content
        cells = [cell for cell in row.find_all(["td", "th"]) if cell.get_text(strip=True)]

        if not cells:
            return None

        # Usually the title is in the first cell with content
        main_cell = cells[0]

        # Try to unwrap to get to the actual styled content
        # Priority: span > div > p
        for tag_name in ["span", "div", "p"]:
            content = main_cell.find(tag_name)
            if content and content.get_text(strip=True):
                return content

        # If no inner tags, return the cell itself
        return main_cell

    def _has_meaningful_content(self, html_tag: HtmlTag) -> bool:
        """
        Check if an element has meaningful content beyond just structure.

        A table has meaningful content if it has:
        - Multiple rows with text
        - OR numbers/data (indicating actual table data)
        - OR substantial text (> 20 chars)

        Args:
            html_tag: The HTML tag to check

        Returns:
            True if the element has meaningful content, False otherwise
        """
        text = html_tag.text.strip()

        # Very short text suggests no meaningful content
        if len(text) < 20:
            return False

        # Check if it's a data table (has numbers)
        has_numbers = any(c.isdigit() for c in text)

        # Count rows with text
        bs4_tag = html_tag._bs4
        if bs4_tag.name == "table":
            table = bs4_tag
        else:
            table = bs4_tag.find("table")

        if table:
            content_rows = [
                row for row in table.find_all("tr")
                if row.get_text(strip=True)
            ]
            has_multiple_rows = len(content_rows) >= 2

            return has_numbers or has_multiple_rows

        # If not a table or can't determine, use text length
        return len(text) > 50