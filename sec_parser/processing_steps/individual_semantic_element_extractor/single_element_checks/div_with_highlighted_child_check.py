from __future__ import annotations

from typing import TYPE_CHECKING

from sec_parser.semantic_elements.highlighted_text_element import TextStyle
from sec_parser.processing_steps.individual_semantic_element_extractor.single_element_checks.abstract_single_element_check import (
    AbstractSingleElementCheck,
)

if TYPE_CHECKING:  # pragma: no cover
    from sec_parser.semantic_elements.abstract_semantic_element import (
        AbstractSemanticElement,
    )


class DivWithHighlightedChildCheck(AbstractSingleElementCheck):
    """
    DivWithHighlightedChildCheck is a check that determines if a `<div>` element
    should be split into multiple elements.
    
    It is designed to handle cases where a `<div>` acts as a container for
    distinct sections, one of which is a highlighted title.
    """

    def contains_single_element(self, element: AbstractSemanticElement) -> bool | None:
        if element.html_tag.name != "div":
            return None

        children = element.html_tag.get_children()
        if not children:
            return True

        # This check is crucial to avoid breaking down paragraphs that contain
        # highlighted words. We only want to split `<div>` elements that are
        # used for structuring content, not for styling inline text.
        if element.html_tag.has_text_outside_tags([child.name for child in children]):
            return True

        highlighted_children = 0
        for child in children:
            # We reuse the TextStyle logic to determine if a child element is highlighted
            style = TextStyle.from_style_and_text(
                child.get_text_styles_metrics(), child.text
            )
            if style:
                highlighted_children += 1

        # If the <div> contains multiple children and at least one of them is
        # highlighted, it's a strong indicator that the <div> is a container
        # for multiple semantic elements and should be split.
        if len(children) > 1 and highlighted_children > 0:
            #print(element.text)
            return False

        return None