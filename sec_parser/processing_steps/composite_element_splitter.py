from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from sec_parser.processing_engine.html_tag import HtmlTag
from sec_parser.processing_steps.abstract_classes.abstract_element_batch_processing_step import (
    AbstractElementBatchProcessingStep,
)
from sec_parser.semantic_elements.abstract_semantic_element import (
    AbstractSemanticElement,
)
from sec_parser.semantic_elements.semantic_elements import NotYetClassifiedElement

if TYPE_CHECKING:  # pragma: no cover
    # Corrected the import from types to processing_context
    from sec_parser.processing_steps.abstract_classes.processing_context import (
        ElementProcessingContext,
    )


class CompositeElementSplitter(AbstractElementBatchProcessingStep):
    """
    CompositeElementSplitter is a processing step that splits composite
    HTML elements into smaller, more semantically coherent units.

    This is crucial for handling cases where a single container tag (like a <div>)
    holds multiple distinct semantic items (like a title and a paragraph) that
    should be classified individually.
    """

    def _process_batch(
        self,
        elements: list[AbstractSemanticElement],
        context: ElementProcessingContext, 
    ) -> list[AbstractSemanticElement]:
        new_elements = []
        for element in elements:
            new_elements.extend(self._split_element_if_needed(element))
        return new_elements

    def _split_element_if_needed(
        self,
        element: AbstractSemanticElement,
    ) -> Iterable[AbstractSemanticElement]:
        """
        Splits an element if it's an unclassified container with multiple
        distinct child elements.
        """
        # We only want to split elements that have not yet been classified
        # and contain multiple children.
        if (
            not isinstance(element, NotYetClassifiedElement)
            or not element.html_tag.has_children()
        ):
            yield element
            return

        # A simple but effective heuristic: if an unclassified element contains
        # child divs or spans, it's likely a composite element that needs splitting.
        # This avoids incorrectly splitting elements that are already semantically meaningful.
        children = list(element.html_tag.get_children())
        if len(children) > 1 and any(c.name in ('div', 'span', 'p') for c in children if isinstance(c, HtmlTag)):
            for child in children:
                # Ensure we are only creating elements from actual tags with content
                if isinstance(child, HtmlTag) and child.get_text(strip=True):
                    yield NotYetClassifiedElement.create_from_element(
                        element,
                        html_tag=child,
                        log_origin=self.__class__.__name__,
                    )
        else:
            yield element