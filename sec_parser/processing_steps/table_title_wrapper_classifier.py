from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from sec_parser.processing_steps.abstract_classes.abstract_elementwise_processing_step import (
    AbstractElementwiseProcessingStep,
    ElementProcessingContext,
)
from sec_parser.semantic_elements.semantic_elements import NotYetClassifiedElement
from sec_parser.semantic_elements.table_element.table_element import TableElement

if TYPE_CHECKING:  # pragma: no cover
    import bs4

    from sec_parser.semantic_elements.abstract_semantic_element import (
        AbstractSemanticElement,
    )


class TableTitleWrapperClassifier(AbstractElementwiseProcessingStep):
    """
    TableTitleWrapperClassifier class for converting table elements that are used
    as styling wrappers for titles into NotYetClassifiedElement instances.
    This allows them to be re-classified as titles in subsequent steps.
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
        

        if self._is_title_wrapper_table(element.html_tag._bs4):
            return NotYetClassifiedElement.create_from_element(
                element,
                log_origin=self.__class__.__name__,
            )

        return element

    def _is_title_wrapper_table(self, table: bs4.Tag, title_element: bs4.Tag = None) -> bool:
        """
        Detect if a table is just a styling wrapper for a title.
        """
        if title_element and title_element.name in ['td', 'th']:
            if title_element.find_parent('table') == table:
                return False

        rows = table.find_all('tr')

        if len(rows) > 5:
            return False

        cells_with_content = 0
        total_text = []

        for row in rows:
            cells = row.find_all(['td', 'th'])
            for cell in cells:
                text = cell.get_text(strip=True)
                if text:
                    cells_with_content += 1
                    total_text.append(text)

        if cells_with_content <= 2:
            combined_text = ' '.join(total_text)
            logger.info(f"Title wrapper check: {cells_with_content} cells, text: '{combined_text[:50]}...'")
            return True

        return False