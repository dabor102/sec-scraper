"""
TableTitleSplitter - Extracts title-like rows from within table elements.

Simple logic:
1. Find first non-empty row
2. If it's a header row → data table, keep as TableElement
3. If it's NOT a header row:
   - If no more rows → title wrapper, extract as TitleElement
   - If next row IS header → split (first row = title, rest = table)
   - If next row is NOT header → keep as TableElement
"""

from __future__ import annotations

import copy
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
from sec_parser.semantic_elements.table_element.table_element import TableElement
from sec_parser.semantic_elements.title_element import TitleElement

from header_parsing import is_header_row

if TYPE_CHECKING:  # pragma: no cover
    from sec_parser.semantic_elements.abstract_semantic_element import (
        AbstractSemanticElement,
    )


class TableTitleSplitter(AbstractElementwiseProcessingStep):
    """
    Extracts title rows from tables using simple, sequential logic.
    
    Algorithm:
    1. Get first non-empty row
    2. Check if it's a header (periods/units/currency)
    3. If header → keep as table
    4. If NOT header → check next non-empty row
       - If no next row → title wrapper
       - If next IS header → split (title + table)
       - If next NOT header → keep as table
    """

    def __init__(
        self,
        *,
        types_to_process: set[type[AbstractSemanticElement]] | None = None,
        types_to_exclude: set[type[AbstractSemanticElement]] | None = None,
    ) -> None:
        super().__init__(
            types_to_process=types_to_process or {TableElement},
            types_to_exclude=types_to_exclude,
        )

    def _process_element(
        self,
        element: AbstractSemanticElement,
        _: ElementProcessingContext,
    ) -> AbstractSemanticElement:
        """Process a table element using simple sequential logic."""
        if not isinstance(element, TableElement):
            return element

        bs4_tag = element.html_tag._bs4
        table = bs4_tag if bs4_tag.name == "table" else bs4_tag.find("table")

        if not table:
            element.processing_log.add_item(
                message="No table tag found in TableElement",
                log_origin=self.__class__.__name__,
            )
            return element

        all_rows = table.find_all("tr")
        if len(all_rows) == 0:
            return element

        # Get non-empty rows with their original indices
        non_empty_rows = [
            (idx, row) for idx, row in enumerate(all_rows)
            if row.get_text(strip=True)
        ]

        if len(non_empty_rows) == 0:
            return element

        # Step 1: Get first non-empty row
        first_idx, first_row = non_empty_rows[0]
        first_text = first_row.get_text(strip=True)

        # Step 2: Check if first row is a header row
        if is_header_row(first_text, context=None):
            # It's a data table, keep as TableElement
            element.processing_log.add_item(
                message="First row is header row → keeping as data table",
                log_origin=self.__class__.__name__,
            )
            return element

        # Step 3: First row is NOT a header row
        # Check if there are more non-empty rows
        if len(non_empty_rows) == 1:
            # Only one row → title wrapper table
            element.processing_log.add_item(
                message="Single non-header row → extracting as title",
                log_origin=self.__class__.__name__,
            )
            return self._extract_as_title(first_row, element)

        # Step 4: Check if NEXT non-empty row is a header row
        second_idx, second_row = non_empty_rows[1]
        second_text = second_row.get_text(strip=True)

        if is_header_row(second_text, context=None):
            # Split: first row = title, rest = table
            element.processing_log.add_item(
                message="First row is title, second row is header → splitting",
                log_origin=self.__class__.__name__,
            )
            return self._split_title_and_table(
                table, all_rows, first_idx, element
            )

        # Neither first nor second row is header → keep as table
        element.processing_log.add_item(
            message="No header pattern detected → keeping as data table",
            log_origin=self.__class__.__name__,
        )
        return element

    def _extract_as_title(
        self,
        row: bs4.Tag,
        source_element: AbstractSemanticElement,
    ) -> TitleElement:
        """Extract a single row as a TitleElement."""
        # Get the main cell content
        cells = row.find_all(["td", "th"])
        
        # Find cell with most text (or largest colspan)
        if len(cells) == 1:
            main_cell = cells[0]
        else:
            # Prefer cells with colspan
            cells_with_colspan = [
                c for c in cells 
                if c.get("colspan") and int(c.get("colspan", 1)) > 1
            ]
            if cells_with_colspan:
                main_cell = max(
                    cells_with_colspan,
                    key=lambda c: int(c.get("colspan", 1))
                )
            else:
                main_cell = max(cells, key=lambda c: len(c.get_text(strip=True)))
        
        title_tag = HtmlTag(main_cell)
        return TitleElement(
            title_tag,
            processing_log=source_element.processing_log.copy(),
            log_origin=self.__class__.__name__,
            level=0,
        )

    def _split_title_and_table(
        self,
        original_table: bs4.Tag,
        all_rows: list[bs4.Tag],
        title_row_idx: int,
        source_element: AbstractSemanticElement,
    ) -> CompositeSemanticElement:
        """Split table into title element and table element."""
        # Extract title from first row
        title_row = all_rows[title_row_idx]
        title_element = self._extract_as_title(title_row, source_element)
        
        # Create new table with remaining rows
        remaining_rows = [
            row for idx, row in enumerate(all_rows) 
            if idx != title_row_idx
        ]
        
        new_table = copy.copy(original_table)
        new_table.clear()
        
        # Preserve table structure
        if original_table.find("tbody"):
            tbody = new_table.new_tag("tbody")
            for row in remaining_rows:
                tbody.append(copy.copy(row))
            new_table.append(tbody)
        else:
            for row in remaining_rows:
                new_table.append(copy.copy(row))
        
        table_tag = HtmlTag(new_table)
        table_element = TableElement(
            table_tag,
            processing_log=source_element.processing_log.copy(),
            log_origin=self.__class__.__name__,
        )
        
        # Return composite with both elements
        return CompositeSemanticElement.create_from_element(
            source_element,
            log_origin=self.__class__.__name__,
            inner_elements=[title_element, table_element],
        )