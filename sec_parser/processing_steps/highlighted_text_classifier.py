from __future__ import annotations

from typing import TYPE_CHECKING

from sec_parser.processing_steps.abstract_classes.abstract_elementwise_processing_step import (
    AbstractElementwiseProcessingStep,
    ElementProcessingContext,
)
from sec_parser.semantic_elements.highlighted_text_element import (
    HighlightedTextElement,
    TextStyle,
)

if TYPE_CHECKING:  # pragma: no cover
    from sec_parser.semantic_elements.abstract_semantic_element import (
        AbstractSemanticElement,
    )


class HighlightedTextClassifier(AbstractElementwiseProcessingStep):
    """
    HighlightedTextClassifier class for converting elements into
    HighlightedTextElement instances.

    This step scans through a list of semantic elements and changes it,
    primarily by replacing suitable candidates with HighlightedTextElement instances.
    """

    def __init__(
        self,
        *,
        types_to_process: set[type[AbstractSemanticElement]] | None = None,
        types_to_exclude: set[type[AbstractSemanticElement]] | None = None,
    ) -> None:
        super().__init__(
            types_to_process=types_to_process,
            types_to_exclude=types_to_exclude,
        )

    def _process_element(
        self,
        element: AbstractSemanticElement,
        _: ElementProcessingContext,
    ) -> AbstractSemanticElement:
        """
        Process a single element and classify it as HighlightedTextElement
        if its style meets the criteria.
        """
        style = self._get_highlight_style(element)
        if style:
            return HighlightedTextElement.create_from_element(
                element,
                log_origin=self.__class__.__name__,
                style=style,
            )
        return element

    def _get_highlight_style(self, element: AbstractSemanticElement) -> TextStyle | None:
        """
        Determine the TextStyle of a semantic element by analyzing its
        underlying HTML tag metrics.
        """
        styles_metrics = element.html_tag.get_text_styles_metrics()
        style = TextStyle.from_style_and_text(styles_metrics, element.text)
        return style if style else None