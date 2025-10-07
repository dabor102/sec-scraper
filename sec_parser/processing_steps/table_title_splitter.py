"""
TableTitleSplitter - Extracts title-like rows from within table elements.

This processing step runs AFTER TableClassifier and identifies rows that look
like titles (based on styling, colspan, position) within tables, extracts them
as TitleElements, and keeps the remaining rows as a TableElement.
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
    Extracts title-like content from the first rows of tables.
    
    This step analyzes TableElements and identifies rows that appear to be titles
    based on:
    - Visual styling (bold, background color, etc.)
    - Structure (large colspan, TH tags)
    - Position (first few rows)
    - Content (not data-heavy)
    
    When title rows are found, the table is split into:
    1. TitleElement(s) for the title rows
    2. TableElement for the remaining data rows
    
    These are wrapped in a CompositeSemanticElement to preserve structure.
    """

    def __init__(
        self,
        *,
        types_to_process: set[type[AbstractSemanticElement]] | None = None,
        types_to_exclude: set[type[AbstractSemanticElement]] | None = None,
        max_title_rows: int = 3,
        min_colspan_for_title: int = 2,
    ) -> None:
        """
        Initialize the TableTitleSplitter.
        
        Args:
            types_to_process: Set of element types to process (default: {TableElement})
            types_to_exclude: Set of element types to exclude from processing
            max_title_rows: Maximum number of rows to check for titles (default: 3)
            min_colspan_for_title: Minimum colspan to consider a cell title-like (default: 2)
        """
        super().__init__(
            types_to_process=types_to_process or {TableElement},
            types_to_exclude=types_to_exclude,
        )
        self._max_title_rows = max_title_rows
        self._min_colspan_for_title = min_colspan_for_title

    def _process_element(
        self,
        element: AbstractSemanticElement,
        _: ElementProcessingContext,
    ) -> AbstractSemanticElement:
        """
        Process a single element to extract title rows from tables.
        
        Args:
            element: The semantic element to process
            _: Processing context (unused)
            
        Returns:
            Either the original element, or a CompositeSemanticElement containing
            extracted titles and the remaining table
        """
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

        rows = table.find_all("tr")
        if len(rows) <= 1:
            # Single row tables are not split
            return element

        title_indices = self._identify_title_rows(rows)

        if not title_indices:
            # No title rows found
            return element

        element.processing_log.add_item(
            message=f"Found {len(title_indices)} title row(s) at indices: {title_indices}",
            log_origin=self.__class__.__name__,
        )

        # Extract title rows and create new elements
        new_elements = []

        for idx in title_indices:
            row = rows[idx]
            title_element = self._create_title_element_from_row(row, element)
            if title_element:
                new_elements.append(title_element)

        # Create remaining table if there are rows left
        remaining_rows = [r for i, r in enumerate(rows) if i not in title_indices]
        if remaining_rows:
            remaining_table = self._create_table_from_rows(table, remaining_rows)
            if self._has_meaningful_content(remaining_table):
                table_tag = HtmlTag(remaining_table)
                table_element = TableElement(
                    table_tag,
                    processing_log=element.processing_log.copy(),
                    log_origin=self.__class__.__name__,
                )
                table_element.processing_log.add_item(
                    message=f"Table reconstructed with {len(remaining_rows)} remaining rows",
                    log_origin=self.__class__.__name__,
                )
                new_elements.append(table_element)

        if not new_elements:
            return element

        # Return composite element containing both titles and table
        return CompositeSemanticElement.create_from_element(
            element,
            log_origin=self.__class__.__name__,
            inner_elements=new_elements,
        )

    def _identify_title_rows(self, rows: list[bs4.Tag]) -> list[int]:
        title_indices = []
        rows_to_check = min(len(rows), self._max_title_rows)
        found_first_title = False

        for idx in range(rows_to_check):
            row = rows[idx]
            
            # Check if row is empty (just spacing)
            if not row.get_text(strip=True):
                if not found_first_title:
                    continue  # Skip initial empty rows
                else:
                    break  # Stop after titles if we hit empty row
            
            if self._is_title_row(row, idx, rows):
                title_indices.append(idx)
                found_first_title = True
            elif found_first_title:
                # Stop after finding titles
                break
        
        return title_indices


    def _is_title_row(
        self,
        row: bs4.Tag,
        row_idx: int,
        all_rows: list[bs4.Tag],
    ) -> bool:
        """
        Determine if a row is a title based on multiple criteria.
        
        Args:
            row: The row to check
            row_idx: Index of the row in the table
            all_rows: All rows in the table (for context)
            
        Returns:
            True if the row appears to be a title, False otherwise
        """
        cells = row.find_all(["td", "th"])
        if not cells:
            return False

        text = row.get_text(strip=True)

        if is_header_row(text, context=None):
            return False  # ← This prevents extracting "2024, 2023, 2022" as title

        # Criterion 1: Contains substantial text (not just numbers/symbols)
        has_words = any(c.isalpha() for c in text) and len(text) > 3
        if not has_words:
            return False

        # Criterion 2: Has title-like styling
        has_large_colspan = any(
            cell.get("colspan") and int(cell.get("colspan", 1)) >= self._min_colspan_for_title
            for cell in cells
        )
        has_bold = row.find(["b", "strong"]) is not None
        has_background = any(
            "background" in str(cell.get("style", "")).lower() for cell in cells
        )
        has_centered = any(
            "center" in str(cell.get("style", "")).lower()
            or str(cell.get("align", "")).lower() == "center"
            for cell in cells
        )
        has_th_tags = any(cell.name == "th" for cell in cells)
        has_font_styling = any(
            "font-weight" in str(cell.get("style", "")).lower()
            or "font-size" in str(cell.get("style", "")).lower()
            for cell in cells
        )

        has_styling = any([
            has_large_colspan,
            has_bold,
            has_background,
            has_centered,
            has_th_tags,
            has_font_styling,
        ])

        if not has_styling:
            return False

        # Criterion 3: Not a typical data row
        cell_texts = [c.get_text(strip=True) for c in cells if c.get_text(strip=True)]
        
        # If there are many cells with content, it's likely a data row
        if len(cell_texts) >= 4:
            # Count numeric cells
            numeric_cells = sum(
                1 for t in cell_texts 
                if any(c.isdigit() for c in t) and len(t) < 20
            )
            # If most cells are numeric, it's a data row
            if numeric_cells >= len(cell_texts) * 0.5:
                return False

        # Criterion 4: Position matters - titles usually in first few rows
        is_early_row = row_idx < self._max_title_rows

        return has_words and has_styling and is_early_row

    def _create_title_element_from_row(
        self,
        row: bs4.Tag,
        source_element: AbstractSemanticElement,
    ) -> TitleElement | None:
        """
        Create a TitleElement from a table row.
        
        Args:
            row: The table row to convert
            source_element: The source element for logging
            
        Returns:
            TitleElement or None if creation fails
        """
        try:
            # Extract the main content from the row
            content = self._extract_row_content(row)
            if not content:
                return None

            title_tag = HtmlTag(content)
            title_element = TitleElement(
                title_tag,
                processing_log=source_element.processing_log.copy(),
                log_origin=self.__class__.__name__,
                level=0,  # Default level, will be adjusted by TitleClassifier
            )
            title_element.processing_log.add_item(
                message="Extracted from table row",
                log_origin=self.__class__.__name__,
            )
            return title_element
        except Exception as e:  # noqa: BLE001
            # Log error but don't fail the entire processing
            source_element.processing_log.add_item(
                message=f"Failed to create title element: {e}",
                log_origin=self.__class__.__name__,
            )
            return None

    def _extract_row_content(self, row: bs4.Tag) -> bs4.Tag | None:
        """
        Extract the main content from a table row.
        
        Args:
            row: The table row
            
        Returns:
            The main content tag or None
        """
        cells = row.find_all(["td", "th"])
        if not cells:
            return None

        # Strategy 1: If there's only one cell, use it
        if len(cells) == 1:
            return cells[0]

        # Strategy 2: Find cell with largest colspan
        cells_with_colspan = [
            c for c in cells if c.get("colspan") and int(c.get("colspan", 1)) > 1
        ]
        if cells_with_colspan:
            # Use the cell with the largest colspan
            main_cell = max(
                cells_with_colspan,
                key=lambda c: int(c.get("colspan", 1)),
            )
            return main_cell

        # Strategy 3: Find cell with most text content
        main_cell = max(cells, key=lambda c: len(c.get_text(strip=True)))
        return main_cell

    def _create_table_from_rows(
        self,
        original_table: bs4.Tag,
        rows: list[bs4.Tag],
    ) -> bs4.Tag:
        """
        Create a new table with specified rows.
        
        Args:
            original_table: The original table tag
            rows: List of rows to include in new table
            
        Returns:
            New table tag with copied rows
        """
        new_table = copy.copy(original_table)
        new_table.clear()

        # Copy table structure elements if they exist
        if original_table.find("thead"):
            # If there was a thead but we removed its rows, don't include it
            pass
        if original_table.find("tbody"):
            tbody = new_table.new_tag("tbody")
            for row in rows:
                tbody.append(copy.copy(row))
            new_table.append(tbody)
        else:
            for row in rows:
                new_table.append(copy.copy(row))

        return new_table

    def _has_meaningful_content(self, table: bs4.Tag) -> bool:
        """
        Check if a table has meaningful content worth preserving.
        
        Args:
            table: The table tag to check
            
        Returns:
            True if the table has meaningful content
        """
        text = table.get_text(strip=True)
        
        # Very short text suggests no meaningful content
        if len(text) < 10:
            return False

        # Check for data indicators
        rows = table.find_all("tr")
        if len(rows) < 1:
            return False

        # If there are numbers, it's likely meaningful data
        has_numbers = any(c.isdigit() for c in text)
        if has_numbers:
            return True

        # If there's substantial text content, keep it
        return len(text) > 50