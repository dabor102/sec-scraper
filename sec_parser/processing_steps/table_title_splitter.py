"""
TableTitleSplitter - Extracts title-like rows from within table elements.
"""

from __future__ import annotations
import copy
from typing import TYPE_CHECKING
import bs4
import logging

logger = logging.getLogger(__name__)

from sec_parser.processing_engine.html_tag import HtmlTag
from sec_parser.processing_steps.abstract_classes.abstract_elementwise_processing_step import (
    AbstractElementwiseProcessingStep,
    ElementProcessingContext,
)
from sec_parser.semantic_elements.composite_semantic_element import (
    CompositeSemanticElement,
)
from sec_parser.semantic_elements.semantic_elements import NotYetClassifiedElement
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
        if not isinstance(element, TableElement):
            return element

        bs4_tag = element.html_tag._bs4
        table = bs4_tag if bs4_tag.name == "table" else bs4_tag.find("table")

        if not table:
            return element

        all_rows = table.find_all("tr", recursive=False)
        if not all_rows:
            return element

        title_row_indices = []
        first_header_idx = -1

        for idx, row in enumerate(all_rows):
            if not row.get_text(strip=True):
                # Keep track of empty rows, but don't classify them as titles
                continue
            text = row.get_text(separator=' ', strip=True)
            if is_header_row(text, context=None):
                first_header_idx = idx
                break
            else:
                title_row_indices.append(idx)
        
        if not title_row_indices:
            return element

        # The table rows start from the first header index.
        # If no header is found, then there are no table rows, only title rows.
        table_rows_indices = []
        if first_header_idx != -1:
            table_rows_indices = list(range(first_header_idx, len(all_rows)))

        return self._split_rows_into_elements(
            all_rows, title_row_indices, table_rows_indices, element
        )

    def _split_rows_into_elements(
        self,
        all_rows: list[bs4.Tag],
        title_row_indices: list[int],
        table_rows_indices: list[int],
        source_element: AbstractSemanticElement,
    ) -> CompositeSemanticElement:
        print(f"Splitting table. Title indices: {title_row_indices}, Table indices: {table_rows_indices}")
        new_elements: list[AbstractSemanticElement] = []
        soup = bs4.BeautifulSoup("", "lxml")

        # Create a separate NotYetClassifiedElement for each title row
        for idx in title_row_indices:
            row_tag = all_rows[idx]
            # Wrap each row in a new table structure to make it a valid, standalone HTML snippet
            new_table_for_row = soup.new_tag("table")
            new_table_for_row.append(copy.copy(row_tag))
            new_element = NotYetClassifiedElement(
                HtmlTag(new_table_for_row),
                processing_log=source_element.processing_log.copy(),
                log_origin=self.__class__.__name__,
            )
            new_elements.append(new_element)
            logger.debug(f"Created NotYetClassifiedElement for title row {idx}")

        # Reconstruct a single new table from the remaining contiguous rows
        if table_rows_indices:
            try:
                new_table_tag = soup.new_tag("table")
                tbody = soup.new_tag("tbody")
                new_table_tag.append(tbody)
                for idx in table_rows_indices:
                    tbody.append(copy.copy(all_rows[idx]))
                
                table_element = TableElement(
                    HtmlTag(new_table_tag),
                    processing_log=source_element.processing_log.copy(),
                    log_origin=self.__class__.__name__,
                )
                new_elements.append(table_element)
                logger.debug("Successfully created new TableElement from remaining rows.")
            except Exception:
                logger.error("Error creating new TableElement.", exc_info=True)
                pass

        return CompositeSemanticElement.create_from_element(
            source_element,
            log_origin=self.__class__.__name__,
            inner_elements=new_elements,
        )