"""
DocumentStyleAnalyzer - Analyzes style distribution in SEC documents.

This module provides statistical analysis of styling properties (font-size,
font-weight, color, etc.) to identify patterns that could help detect titles
and other semantic elements dynamically.

Usage:
    from sec_parser.utils.document_style_analyzer import DocumentStyleAnalyzer
    
    analyzer = DocumentStyleAnalyzer()
    results = analyzer.analyze(html)
    analyzer.print_summary()
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import bs4

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class StylePropertyStats:
    """Statistics for a single style property-value pair."""
    value: str
    char_count: int
    occurrence_count: int
    percentage: float = 0.0
    avg_text_length: float = 0.0
    min_text_length: int = 0
    max_text_length: int = 0


@dataclass
class StyleCombinationStats:
    """Statistics for a combination of style properties."""
    properties: frozenset[tuple[str, str]]
    char_count: int
    occurrence_count: int
    percentage: float = 0.0
    avg_text_length: float = 0.0
    sample_texts: list[str] = field(default_factory=list)


@dataclass
class DocumentStyleAnalysis:
    """Complete style analysis results for a document."""
    total_chars: int
    total_text_segments: int
    
    # Individual property statistics
    font_size_stats: dict[str, StylePropertyStats]
    font_weight_stats: dict[str, StylePropertyStats]
    color_stats: dict[str, StylePropertyStats]
    font_family_stats: dict[str, StylePropertyStats]
    text_align_stats: dict[str, StylePropertyStats]
    text_transform_stats: dict[str, StylePropertyStats]
    
    # Style combination statistics
    style_combinations: list[StyleCombinationStats]
    
    # Metadata
    has_styling: bool = True


class DocumentStyleAnalyzer:
    """
    Analyzes the distribution of styling properties in SEC documents.
    
    This class performs a comprehensive statistical analysis of CSS styles
    applied to text content, helping identify patterns that distinguish
    titles from regular text.
    
    Key Features:
    - Analyzes individual style properties (font-size, color, etc.)
    - Tracks style combinations that appear together
    - Calculates frequency and percentage distributions
    - Identifies rare/distinctive styles
    - Associates styles with text length patterns
    
    Example:
        >>> analyzer = DocumentStyleAnalyzer()
        >>> results = analyzer.analyze(html_content)
        >>> analyzer.print_summary()
        
        Document Style Analysis Summary
        ================================
        Total characters: 125,430
        Total text segments: 1,247
        
        Font Size Distribution:
        -----------------------
        10pt: 85.2% (106,866 chars, 1,105 occurrences)
        14pt: 8.1% (10,160 chars, 89 occurrences)
        18pt: 3.2% (4,014 chars, 34 occurrences)
        ...
    """
    
    def __init__(self) -> None:
        self._results: DocumentStyleAnalysis | None = None
        self._raw_data: dict = {}
    
    def analyze(self, html: str | bytes) -> DocumentStyleAnalysis:
        """
        Analyze the style distribution in the given HTML document.
        
        Args:
            html: HTML content as string or bytes
            
        Returns:
            DocumentStyleAnalysis containing all statistical results
        """
        soup = bs4.BeautifulSoup(html, features="lxml")
        
        # Initialize data collectors
        total_chars = 0
        total_segments = 0
        
        # Track individual properties
        property_data = {
            'font-size': defaultdict(lambda: {'chars': 0, 'count': 0, 'lengths': []}),
            'font-weight': defaultdict(lambda: {'chars': 0, 'count': 0, 'lengths': []}),
            'color': defaultdict(lambda: {'chars': 0, 'count': 0, 'lengths': []}),
            'font-family': defaultdict(lambda: {'chars': 0, 'count': 0, 'lengths': []}),
            'text-align': defaultdict(lambda: {'chars': 0, 'count': 0, 'lengths': []}),
            'text-transform': defaultdict(lambda: {'chars': 0, 'count': 0, 'lengths': []}),
        }
        
        # Track style combinations
        combination_data = defaultdict(lambda: {
            'chars': 0,
            'count': 0,
            'lengths': [],
            'samples': []
        })
        
        # Iterate through all text nodes
        for text_node in soup.find_all(string=True, recursive=True):
            text = text_node.strip()
            char_count = len(text)
            
            if char_count == 0:
                continue
            
            total_chars += char_count
            total_segments += 1
            
            # Get effective styles for this text node
            parent = text_node.find_parent()
            effective_styles = self._compute_effective_style(parent)
            
            # Track individual properties
            for prop in property_data.keys():
                value = effective_styles.get(prop, 'default')
                property_data[prop][value]['chars'] += char_count
                property_data[prop][value]['count'] += 1
                property_data[prop][value]['lengths'].append(char_count)
            
            # Track style combinations
            style_combo = frozenset(effective_styles.items())
            combination_data[style_combo]['chars'] += char_count
            combination_data[style_combo]['count'] += 1
            combination_data[style_combo]['lengths'].append(char_count)
            
            # Store sample texts (limit to 3 samples per combination)
            if len(combination_data[style_combo]['samples']) < 3:
                sample = text[:100] + "..." if len(text) > 100 else text
                combination_data[style_combo]['samples'].append(sample)
        
        # Convert raw data to statistics objects
        has_styling = total_chars > 0
        
        font_size_stats = self._create_property_stats(
            property_data['font-size'], total_chars
        )
        font_weight_stats = self._create_property_stats(
            property_data['font-weight'], total_chars
        )
        color_stats = self._create_property_stats(
            property_data['color'], total_chars
        )
        font_family_stats = self._create_property_stats(
            property_data['font-family'], total_chars
        )
        text_align_stats = self._create_property_stats(
            property_data['text-align'], total_chars
        )
        text_transform_stats = self._create_property_stats(
            property_data['text-transform'], total_chars
        )
        
        style_combinations = self._create_combination_stats(
            combination_data, total_chars
        )
        
        # Store raw data for debugging
        self._raw_data = {
            'property_data': property_data,
            'combination_data': combination_data,
        }
        
        self._results = DocumentStyleAnalysis(
            total_chars=total_chars,
            total_text_segments=total_segments,
            font_size_stats=font_size_stats,
            font_weight_stats=font_weight_stats,
            color_stats=color_stats,
            font_family_stats=font_family_stats,
            text_align_stats=text_align_stats,
            text_transform_stats=text_transform_stats,
            style_combinations=style_combinations,
            has_styling=has_styling,
        )
        
        return self._results
    
    def _compute_effective_style(self, tag: bs4.Tag | None) -> dict[str, str]:
        """
        Compute the effective styles for a tag by traversing up the parent hierarchy.
        """
        effective_styles: dict[str, str] = {}
        current_tag = tag
        
        while current_tag:
            if isinstance(current_tag, bs4.Tag) and "style" in current_tag.attrs:
                styles = current_tag["style"]
                if isinstance(styles, str):
                    for style in styles.split(";"):
                        if ":" in style:
                            prop, val = style.split(":", 1)
                            prop = prop.strip()
                            val = val.strip()
                            # Only set if not previously set (CSS cascading)
                            effective_styles.setdefault(prop, val)
            
            current_tag = current_tag.find_parent() if isinstance(current_tag, bs4.Tag) else None
        
        return effective_styles
    
    def _create_property_stats(
        self,
        raw_data: dict,
        total_chars: int,
    ) -> dict[str, StylePropertyStats]:
        """Convert raw property data to StylePropertyStats objects."""
        stats = {}
        
        for value, data in raw_data.items():
            char_count = data['chars']
            occurrence_count = data['count']
            lengths = data['lengths']
            
            stats[value] = StylePropertyStats(
                value=value,
                char_count=char_count,
                occurrence_count=occurrence_count,
                percentage=(char_count / total_chars * 100) if total_chars > 0 else 0,
                avg_text_length=sum(lengths) / len(lengths) if lengths else 0,
                min_text_length=min(lengths) if lengths else 0,
                max_text_length=max(lengths) if lengths else 0,
            )
        
        return stats
    
    def _create_combination_stats(
        self,
        raw_data: dict,
        total_chars: int,
    ) -> list[StyleCombinationStats]:
        """Convert raw combination data to StyleCombinationStats objects."""
        stats = []
        
        for style_combo, data in raw_data.items():
            char_count = data['chars']
            occurrence_count = data['count']
            lengths = data['lengths']
            samples = data['samples']
            
            stats.append(StyleCombinationStats(
                properties=style_combo,
                char_count=char_count,
                occurrence_count=occurrence_count,
                percentage=(char_count / total_chars * 100) if total_chars > 0 else 0,
                avg_text_length=sum(lengths) / len(lengths) if lengths else 0,
                sample_texts=samples,
            ))
        
        # Sort by percentage (descending)
        stats.sort(key=lambda x: x.percentage, reverse=True)
        
        return stats
    
    def print_summary(
        self,
        *,
        top_n: int = 10,
        min_percentage: float = 0.5,
        show_combinations: bool = True,
    ) -> None:
        """
        Print a human-readable summary of the style analysis.
        
        Args:
            top_n: Number of top items to show for each category
            min_percentage: Minimum percentage threshold for displaying items
            show_combinations: Whether to show style combination analysis
        """
        if self._results is None:
            print("No analysis results available. Run analyze() first.")
            return
        
        r = self._results
        
        print("\n" + "=" * 80)
        print("DOCUMENT STYLE ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"\nTotal characters analyzed: {r.total_chars:,}")
        print(f"Total text segments: {r.total_text_segments:,}")
        print(f"Has styling: {r.has_styling}")
        
        # Font Size Distribution
        print("\n" + "-" * 80)
        print("FONT SIZE DISTRIBUTION")
        print("-" * 80)
        self._print_property_stats(r.font_size_stats, top_n, min_percentage)
        
        # Font Weight Distribution
        print("\n" + "-" * 80)
        print("FONT WEIGHT DISTRIBUTION")
        print("-" * 80)
        self._print_property_stats(r.font_weight_stats, top_n, min_percentage)
        
        # Color Distribution
        print("\n" + "-" * 80)
        print("COLOR DISTRIBUTION")
        print("-" * 80)
        self._print_property_stats(r.color_stats, top_n, min_percentage)
        
        # Font Family Distribution
        print("\n" + "-" * 80)
        print("FONT FAMILY DISTRIBUTION")
        print("-" * 80)
        self._print_property_stats(r.font_family_stats, top_n, min_percentage)
        
        # Text Align Distribution
        print("\n" + "-" * 80)
        print("TEXT ALIGN DISTRIBUTION")
        print("-" * 80)
        self._print_property_stats(r.text_align_stats, top_n, min_percentage)
        
        # Text Transform Distribution
        print("\n" + "-" * 80)
        print("TEXT TRANSFORM DISTRIBUTION")
        print("-" * 80)
        self._print_property_stats(r.text_transform_stats, top_n, min_percentage)
        
        # Style Combinations (rare ones that might be titles)
        if show_combinations:
            print("\n" + "-" * 80)
            print("RARE STYLE COMBINATIONS (Potential Titles)")
            print("-" * 80)
            print("(Showing combinations with < 5% of total text)")
            self._print_rare_combinations(r.style_combinations, max_percentage=5.0)
        
        print("\n" + "=" * 80)
    
    def _print_property_stats(
        self,
        stats: dict[str, StylePropertyStats],
        top_n: int,
        min_percentage: float,
    ) -> None:
        """Print statistics for a single property."""
        # Sort by percentage descending
        sorted_stats = sorted(
            stats.values(),
            key=lambda x: x.percentage,
            reverse=True,
        )
        
        # Filter and limit
        filtered = [s for s in sorted_stats if s.percentage >= min_percentage][:top_n]
        
        if not filtered:
            print("  No significant values found")
            return
        
        for stat in filtered:
            print(f"\n  {stat.value}:")
            print(f"    Percentage: {stat.percentage:.2f}%")
            print(f"    Characters: {stat.char_count:,}")
            print(f"    Occurrences: {stat.occurrence_count:,}")
            print(f"    Avg text length: {stat.avg_text_length:.1f} chars")
            print(f"    Range: {stat.min_text_length}-{stat.max_text_length} chars")
    
    def _print_rare_combinations(
        self,
        combinations: list[StyleCombinationStats],
        max_percentage: float,
    ) -> None:
        """Print rare style combinations that might indicate titles."""
        rare = [c for c in combinations if c.percentage < max_percentage]
        
        if not rare:
            print("  No rare combinations found")
            return
        
        for i, combo in enumerate(rare[:15], 1):  # Show top 15 rare combinations
            print(f"\n  Combination #{i}:")
            print(f"    Percentage: {combo.percentage:.2f}%")
            print(f"    Occurrences: {combo.occurrence_count}")
            print(f"    Avg text length: {combo.avg_text_length:.1f} chars")
            
            # Show the styles
            print("    Styles:")
            for prop, value in sorted(combo.properties):
                print(f"      {prop}: {value}")
            
            # Show sample texts
            if combo.sample_texts:
                print("    Sample texts:")
                for sample in combo.sample_texts:
                    print(f"      - \"{sample}\"")
    
    def get_results(self) -> DocumentStyleAnalysis | None:
        """Get the analysis results."""
        return self._results
    
    def get_rare_styles(
        self,
        property_name: str,
        max_percentage: float = 10.0,
    ) -> list[StylePropertyStats]:
        """
        Get rare style values for a specific property that might indicate titles.
        
        Args:
            property_name: Style property to analyze ('font-size', 'color', etc.)
            max_percentage: Maximum percentage to consider "rare"
            
        Returns:
            List of rare StylePropertyStats sorted by percentage ascending
        """
        if self._results is None:
            return []
        
        property_map = {
            'font-size': self._results.font_size_stats,
            'font-weight': self._results.font_weight_stats,
            'color': self._results.color_stats,
            'font-family': self._results.font_family_stats,
            'text-align': self._results.text_align_stats,
            'text-transform': self._results.text_transform_stats,
        }
        
        stats = property_map.get(property_name)
        if not stats:
            return []
        
        rare = [s for s in stats.values() if s.percentage < max_percentage]
        rare.sort(key=lambda x: x.percentage)
        
        return rare