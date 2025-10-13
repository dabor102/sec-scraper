from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from sec_parser.exceptions import SecParserValueError
from sec_parser.semantic_elements.abstract_semantic_element import (
    AbstractSemanticElement,
)
from sec_parser.utils.py_utils import exceeds_capitalization_threshold

if TYPE_CHECKING:  # pragma: no cover
    from sec_parser.processing_engine.html_tag import HtmlTag
    from sec_parser.processing_engine.processing_log import LogItemOrigin, ProcessingLog


class HighlightedTextElement(AbstractSemanticElement):
    """
    The HighlightedTextElement class, among other uses,
    is an intermediate step in identifying title elements.

    For example:
    ============
    First, elements with specific styles (like bold or italic text)
    are classified as HighlightedTextElements.
    These are later examined to determine if they should
    be considered TitleElements.
    """

    def __init__(
        self,
        html_tag: HtmlTag,
        *,
        processing_log: ProcessingLog | None = None,
        style: TextStyle | None = None,
        log_origin: LogItemOrigin | None = None,
    ) -> None:
        super().__init__(html_tag, processing_log=processing_log, log_origin=None)
        if style is None:
            msg = "styles must be specified for HighlightedElement"
            raise SecParserValueError(msg)
        self.style = style
        self.log_init(log_origin)

    @classmethod
    def create_from_element(
        cls,
        source: AbstractSemanticElement,
        log_origin: LogItemOrigin,
        *,
        style: TextStyle | None = None,
    ) -> HighlightedTextElement:
        if style is None:
            msg = "Style must be provided."
            raise SecParserValueError(msg)
        return cls(
            source.html_tag,
            style=style,
            processing_log=source.processing_log,
            log_origin=log_origin,
        )

    def to_dict(
        self,
        *,
        include_previews: bool = False,
        include_contents: bool = False,
    ) -> dict[str, Any]:
        return {
            **super().to_dict(
                include_previews=include_previews,
                include_contents=include_contents,
            ),
            "text_style": asdict(self.style),
        }


@dataclass(frozen=True)
class TextStyle:
    PERCENTAGE_THRESHOLD = 80
    BOLD_THRESHOLD = 600

    is_all_uppercase: bool = False
    bold_with_font_weight: bool = False
    italic: bool = False
    centered: bool = False
    underline: bool = False
    has_color: bool = False

    def __bool__(self) -> bool:
        return any(asdict(self).values())

    @classmethod
    def from_style_and_text(
        cls,
        style_percentage: dict[tuple[str, str], float],
        text: str,
        dominant_colors: set[str] | None = None,
    ) -> TextStyle:
        # Text checks
        is_all_uppercase = exceeds_capitalization_threshold(
            text,
            cls.PERCENTAGE_THRESHOLD,
        )

        # Filter styles that meet the percentage threshold
        filtered_styles = {
            (k, v): p
            for (k, v), p in style_percentage.items()
            if p >= cls.PERCENTAGE_THRESHOLD
        }

        # Define checks for each style
        style_checks = {
            "bold_with_font_weight": cls._is_bold_with_font_weight,
            "italic": lambda k, v: k == "font-style" and v == "italic",
            "centered": lambda k, v: k == "text-align" and v == "center",
            "underline": lambda k, v: k == "text-decoration" and v == "underline",
        }

        # Apply checks to the filtered styles
        style_results = {
            style: any(check(k, v) for (k, v) in filtered_styles)
            for style, check in style_checks.items()
        }

        # Check for non-dominant color
        has_color = cls._has_non_dominant_color(filtered_styles, dominant_colors)

        # Return a TextStyle instance with the results
        return cls(
            is_all_uppercase=is_all_uppercase,
            bold_with_font_weight=style_results["bold_with_font_weight"],
            italic=style_results["italic"],
            centered=style_results["centered"],
            underline=style_results["underline"],
            has_color=has_color,
        )

    @classmethod
    def _is_bold_with_font_weight(cls, key: str, value: str) -> bool:
        if key != "font-weight":
            return False
        if value == "bold":
            return True
        try:
            return int(value) >= cls.BOLD_THRESHOLD
        except ValueError:
            return False

    @classmethod
    def _has_non_dominant_color(
        cls,
        filtered_styles: dict[tuple[str, str], float],
        dominant_colors: set[str] | None,
    ) -> bool:
        """
        Check if the element has a color different from the dominant colors.
        
        Args:
            filtered_styles: Styles that meet the percentage threshold
            dominant_colors: Set of dominant colors in the document (e.g., {'#000000', '#262626'})
        
        Returns:
            True if element has a non-dominant color, False otherwise
        """
        if not dominant_colors:
            return False
        
        # Normalize dominant colors for comparison
        normalized_dominant = {color.lower().strip() for color in dominant_colors}
        
        for (key, value), _ in filtered_styles.items():
            if key == "color":
                element_color = value.lower().strip()
                if element_color not in normalized_dominant:
                    return True
        
        return False