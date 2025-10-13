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
    
    Uses a two-pass approach:
    - Iteration 0: Collect color statistics from all elements
    - Iteration 1: Classify elements using the determined dominant colors
    """

    _NUM_ITERATIONS = 2
    DOMINANT_COLOR_THRESHOLD = 5.0  # Colors with ≥5% of text are considered dominant

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
        self._color_char_counts: dict[str, float] = {}
        self._dominant_colors: set[str] = set()

    def _process_element(
        self,
        element: AbstractSemanticElement,
        context: ElementProcessingContext,
    ) -> AbstractSemanticElement:
        """
        Process a single element and classify it as HighlightedTextElement
        if its style meets the criteria.
        """
        if context.iteration == 0:
            self._collect_color_stats(element)
            return element
        
        if context.iteration == 1:
            if not self._dominant_colors:
                self._determine_dominant_colors()
            
            style = self._get_highlight_style(element)
            if style:
                return HighlightedTextElement.create_from_element(
                    element,
                    log_origin=self.__class__.__name__,
                    style=style,
                )
            return element
        
        msg = f"Invalid iteration: {context.iteration}"
        raise ValueError(msg)

    def _collect_color_stats(self, element: AbstractSemanticElement) -> None:
        """Collect color statistics from element for dominant color calculation."""
        styles_metrics = element.html_tag.get_text_styles_metrics()
        
        for (prop, value), percentage in styles_metrics.items():
            if prop == "color":
                char_count = len(element.text) * (percentage / 100.0)
                normalized_color = value.lower().strip()
                
                if normalized_color in self._color_char_counts:
                    self._color_char_counts[normalized_color] += char_count
                else:
                    self._color_char_counts[normalized_color] = char_count

    def _determine_dominant_colors(self) -> None:
        """
        Determine the dominant colors based on collected statistics.
        Any color representing >= 5% of total text is considered dominant.
        """
        if not self._color_char_counts:
            self._dominant_colors = set()
            return
        
        total_chars = sum(self._color_char_counts.values())
        
        # Find all colors that exceed the threshold
        for color, char_count in self._color_char_counts.items():
            percentage = (char_count / total_chars * 100) if total_chars > 0 else 0
            if percentage >= self.DOMINANT_COLOR_THRESHOLD:
                self._dominant_colors.add(color)

    def _get_highlight_style(self, element: AbstractSemanticElement) -> TextStyle | None:
        """
        Determine the TextStyle of a semantic element by analyzing its
        underlying HTML tag metrics.
        """
        styles_metrics = element.html_tag.get_text_styles_metrics()
        style = TextStyle.from_style_and_text(
            styles_metrics,
            element.text,
            dominant_colors=self._dominant_colors,
        )
        return style if style else None