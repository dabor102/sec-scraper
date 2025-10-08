from __future__ import annotations

from typing import TYPE_CHECKING

from sec_parser.processing_steps.abstract_classes.abstract_elementwise_processing_step import (
    AbstractElementwiseProcessingStep,
)
from sec_parser.semantic_elements.composite_semantic_element import (
    CompositeSemanticElement,
)

if TYPE_CHECKING:
    from sec_parser.processing_engine.types import ElementwiseProcessingContext
    from sec_parser.semantic_elements.abstract_semantic_element import (
        AbstractSemanticElement,
    )


class CompositeElementSplitter(AbstractElementwiseProcessingStep):
    def __init__(
        self,
        *,
        types_to_process: set[type[CompositeSemanticElement]] | None = None,
        types_to_ignore: set[type[CompositeSemanticElement]] | None = None,
    ) -> None:
        super().__init__(types_to_process=types_to_process)
        self._types_to_ignore = types_to_ignore or set()

    def _process_element(
        self,
        element: AbstractSemanticElement,
        context: ElementwiseProcessingContext,
    ) -> list[AbstractSemanticElement]:
        del context  # unused
        if not isinstance(element, CompositeSemanticElement):
            return [element]
        if any(isinstance(element, t) for t in self._types_to_ignore):
            return [element]

        result = []
        for inner_element in element.inner_elements:
            result.extend(self._process_element(inner_element, context))
        return result