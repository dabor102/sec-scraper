from __future__ import annotations

from typing import TYPE_CHECKING

from sec_parser.processing_steps.abstract_classes.abstract_element_batch_processing_step import (
    AbstractElementBatchProcessingStep,
)
from sec_parser.semantic_elements.composite_semantic_element import (
    CompositeSemanticElement,
)

if TYPE_CHECKING:
    from sec_parser.processing_steps.abstract_classes.processing_context import (
        ElementProcessingContext,
    )
    from sec_parser.semantic_elements.abstract_semantic_element import (
        AbstractSemanticElement,
    )


class CompositeElementSplitter(AbstractElementBatchProcessingStep):
    def __init__(
        self,
        *,
        types_to_process: set[type[CompositeSemanticElement]] | None = None,
        types_to_ignore: set[type[CompositeSemanticElement]] | None = None,
    ) -> None:
        super().__init__()
        self._types_to_process = types_to_process or {CompositeSemanticElement}
        self._types_to_ignore = types_to_ignore or set()

    def _process_elements(
        self,
        elements: list[AbstractSemanticElement],
        context: ElementProcessingContext,
    ) -> list[AbstractSemanticElement]:
        result: list[AbstractSemanticElement] = []
        for element in elements:
            if (
                isinstance(element, CompositeSemanticElement)
                and any(isinstance(element, t) for t in self._types_to_process)
                and not any(isinstance(element, t) for t in self._types_to_ignore)
            ):
                # Recursively process the inner elements and add them to the result
                result.extend(
                    self._process_elements(list(element.inner_elements), context)
                )
            else:
                result.append(element)
        return result