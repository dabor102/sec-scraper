"""
Statement finder using semantic tree navigation.
Locates financial statements in SEC filings using sec-parser.
"""

import logging
import re
from typing import List, Dict, Optional, Tuple, Set
from bs4 import Tag, BeautifulSoup

from sec_parser.processing_engine.core import Edgar10KParser, Edgar10QParser
from sec_parser.semantic_tree.tree_builder import TreeBuilder
from sec_parser.semantic_elements.title_element import TitleElement
from sec_parser.semantic_elements.top_section_title import TopSectionTitle
from sec_parser.semantic_elements.table_element.table_element import TableElement
from sec_parser.semantic_elements.composite_semantic_element import CompositeSemanticElement

from .config import Patterns, StatementKeywords
from .table_processor import normalize_text

logger = logging.getLogger(__name__)


# ============================================================================
# DIAGNOSTIC UTILITIES
# ============================================================================

def dump_html_context(element: Tag, context_lines: int = 10) -> str:
    """
    Dump HTML context around an element for debugging.
    
    Args:
        element: BeautifulSoup element
        context_lines: Number of sibling elements to show before/after
        
    Returns:
        Formatted string showing HTML structure
    """
    lines = []
    lines.append("\n" + "="*80)
    lines.append("HTML CONTEXT DUMP")
    lines.append("="*80)
    
    # Show parent chain
    lines.append("\nParent chain:")
    current = element
    depth = 0
    while current and depth < 5:
        tag_info = f"<{current.name}>"
        if current.get('id'):
            tag_info += f" id='{current.get('id')}'"
        if current.get('class'):
            tag_info += f" class='{' '.join(current.get('class'))}'"
        lines.append(f"  {'  '*depth}{tag_info}")
        current = current.parent
        depth += 1
    
    # Show siblings
    lines.append(f"\nSiblings (showing {context_lines} before and after):")
    if element.parent:
        siblings = list(element.parent.children)
        try:
            current_idx = siblings.index(element)
            start_idx = max(0, current_idx - context_lines)
            end_idx = min(len(siblings), current_idx + context_lines + 1)
            
            for idx in range(start_idx, end_idx):
                sibling = siblings[idx]
                if hasattr(sibling, 'name'):
                    marker = " >>> THIS <<<" if idx == current_idx else ""
                    preview = sibling.get_text(strip=True)[:50]
                    lines.append(f"  [{idx}] <{sibling.name}> {preview}...{marker}")
        except ValueError:
            lines.append("  (Could not determine element position)")
    
    # Show next tables
    lines.append("\nNext tables found:")
    next_table = element.find_next('table')
    table_count = 0
    while next_table and table_count < 5:
        table_count += 1
        rows = len(next_table.find_all('tr'))
        preview = next_table.get_text(strip=True)[:80]
        lines.append(f"  Table #{table_count}: {rows} rows, preview: {preview}...")
        next_table = next_table.find_next('table')
    
    if table_count == 0:
        lines.append("  (No tables found after element)")
    
    lines.append("="*80 + "\n")
    return "\n".join(lines)


# ============================================================================
# SEMANTIC TREE PARSING
# ============================================================================

class StatementFinder:
    """Finds financial statements in SEC filings using semantic tree."""
    
    def __init__(self, form_type: str):
        """
        Initialize statement finder.
        
        Args:
            form_type: '10-Q' or '10-K'
        """
        self.form_type = form_type
        if form_type == '10-K':
            self.parser = Edgar10KParser()
        elif form_type == '10-Q':
            self.parser = Edgar10QParser()
        else:
            raise ValueError(f"Unsupported form type: {form_type}")
    
    def parse_document(self, html_content: str) -> Tuple[Optional[list], Optional[list]]:
        """
        Parse document using semantic parser.
        
        Args:
            html_content: HTML string of the filing
            
        Returns:
            (semantic_tree_nodes, elements) or (None, None) on failure
        """
        try:
            logger.info(f"Parsing document with {self.form_type} parser")
            elements = self.parser.parse(html_content)
            logger.debug(f"Parsed {len(elements)} semantic elements")
            
            tree = TreeBuilder().build(elements)
            semantic_tree_nodes = list(tree.nodes)
            
            logger.info(f"Built semantic tree with {len(semantic_tree_nodes)} nodes")
            return semantic_tree_nodes, elements
            
        except Exception as e:
            logger.error(f"Failed to parse document: {e}")
            return None, None
    
    def find_item8_node(self, tree: list) -> Tuple[int, Optional[TopSectionTitle]]:
        """
        Find 'Item 8' node in semantic tree (for 10-K filings).
        
        Returns:
            (index, node) or (-1, None) if not found
        """
        candidates = []
        
        for i, node in enumerate(tree):
            if isinstance(node.semantic_element, (TopSectionTitle, TitleElement)):
                node_text = normalize_text(node.text)
                if Patterns.ITEM_8.search(node_text):
                    # Look ahead to confirm it's about financial statements
                    look_ahead_text = " ".join(
                        normalize_text(next_node.text) for next_node in tree[i+1:i+6]
                    )
                    is_confirmed = bool(Patterns.FINANCIAL_STATEMENTS.search(look_ahead_text))
                    
                    candidates.append({
                        "index": i,
                        "node": node,
                        "is_confirmed": is_confirmed,
                        "is_top_section": isinstance(node.semantic_element, TopSectionTitle)
                    })
        
        if not candidates:
            logger.warning("Could not find 'Item 8' node")
            return -1, None
        
        # Sort by: confirmed first, TopSectionTitle first, last occurrence first
        candidates.sort(key=lambda c: (
            0 if c['is_confirmed'] else 1,
            0 if c['is_top_section'] else 1,
            -c['index']
        ))
        
        best = candidates[0]
        logger.info(f"Found 'Item 8' at index {best['index']} "
                   f"(Confirmed: {best['is_confirmed']})")
        
        return best['index'], best['node']
    
    def find_statement_titles(
        self,
        tree: list,
        keywords: StatementKeywords
    ) -> List[Dict]:
        """
        Find TitleElements matching statement keywords.
        
        Args:
            tree: Semantic tree nodes
            keywords: StatementKeywords object with must_have/nice_to_have/penalties
            
        Returns:
            List of dicts with 'text' and 'node' keys
        """
        matching_titles = []
        
        logger.debug(f"Searching for must-have keywords: {keywords.must_have}")
        logger.debug(f"Nice-to-have keywords: {keywords.nice_to_have}")
        
        for node in tree:
            if isinstance(node.semantic_element, TitleElement):
                title_text = normalize_text(node.text)
                title_lower = title_text.lower()
                
                # Must have at least one required keyword
                has_required = any(kw in title_lower for kw in keywords.must_have)
                if not has_required:
                    continue
                
                has_nice_to_have = any(kw in title_lower for kw in keywords.nice_to_have)
                
                if has_required or has_nice_to_have:
                    logger.info(f"Found matching title: '{title_text}'")
                    matching_titles.append({
                        'text': title_text,
                        'node': node
                    })
        
        return matching_titles
    
    def score_title(self, title_text: str, keywords: StatementKeywords) -> int:
        """
        Score title based on keyword matching.
        
        Args:
            title_text: Title text to score
            keywords: StatementKeywords with must_have/nice_to_have/penalties
            
        Returns:
            Score (higher is better)
        """
        text = title_text.lower()
        score = 0
        
        # Must-have keywords
        must_have_matches = sum(1 for kw in keywords.must_have if kw in text)
        score += must_have_matches * 100
        
        # Nice-to-have keywords
        nice_to_have_matches = sum(1 for kw in keywords.nice_to_have if kw in text)
        score += nice_to_have_matches * 10
        
        # Penalties
        penalty_matches = sum(1 for kw in keywords.penalties if kw in text)
        score -= penalty_matches * 30
        
        # Penalize numbered items (likely notes)
        if re.match(r'^\s*\d{1,2}\s*[\.\)]', text):
            score -= 50
        
        # Slight penalty for very long titles
        score -= len(text) // 20
        
        return score
    
    def find_statement_table(
        self,
        semantic_tree: list,
        section_name: str,
        keywords: StatementKeywords,
        processed_tables: Set[Tag],
        used_title_texts: Set[str],
        full_soup: Optional[BeautifulSoup] = None
    ) -> Optional[Tuple[Tag, str, Dict]]:
        """
        Find financial statement table using semantic tree.
        
        Args:
            semantic_tree: Full semantic tree
            section_name: Name of statement (e.g., 'Income Statement')
            keywords: StatementKeywords for this statement type
            processed_tables: Set of already processed tables
            used_title_texts: Set of already used title texts
            full_soup: Full document BeautifulSoup (for finding elements in context)
            
        Returns:
            (table_bs4, title_text, metadata) or None
        """
        logger.info(f"\n--- Finding {section_name} (Semantic Tree Method) ---")
        
        search_tree = semantic_tree
        
        # For 10-K, restrict search to after Item 8
        if self.form_type == '10-K':
            item8_index, _ = self.find_item8_node(semantic_tree)
            if item8_index != -1:
                search_tree = semantic_tree[item8_index:]
                logger.info(f"Restricted search to {len(search_tree)} nodes after Item 8")
        
        matching_titles = self.find_statement_titles(search_tree, keywords)
        
        # Filter out already-used titles
        unprocessed_titles = [t for t in matching_titles if t['text'] not in used_title_texts]
        
        if not unprocessed_titles:
            logger.warning(f"No new titles found for {section_name}")
            return None
        
        # Score and prioritize
        prioritized_titles = sorted(
            unprocessed_titles,
            key=lambda t: self.score_title(t['text'], keywords),
            reverse=True
        )
        
        # Try each candidate
        for title_info in prioritized_titles:
            title_text = title_info['text']
            title_node = title_info['node']
            
            logger.info(f"Processing: '{title_text}'")
            
            result = self._extract_table_from_semantic_node(title_node, full_soup)
            
            if not result:
                logger.warning(f"Could not find table for: '{title_text}'")
                continue
            
            table_bs4, metadata = result
            
            # Check if already processed
            if table_bs4 in processed_tables:
                logger.warning("Table already processed, trying next candidate")
                continue
            
            rows = len(table_bs4.find_all('tr'))
            
            # Skip very large tables in 10-K (likely ToC)
            if self.form_type == '10-K' and rows > 100:
                logger.warning(f"Skipping very large table ({rows} rows)")
                continue
            
            logger.info(f"✓ Found table for {section_name}")
            logger.info(f"  Source: {metadata['source']}")
            logger.info(f"  Extracted title: {metadata['is_extracted_title']}")
            logger.info(f"  Rows: {rows}")
            
            return table_bs4, title_text, metadata
        
        logger.error(f"Could not find table for {section_name}")
        return None
    
    def _extract_table_from_semantic_node(
        self, 
        title_node,
        full_soup: Optional[BeautifulSoup] = None
    ) -> Optional[Tuple[Tag, Dict]]:
        """
        Extract table directly using semantic tree references.
        
        Args:
            title_node: Semantic tree node for the title
            full_soup: Full document BeautifulSoup (for Strategy 3)
        
        Returns:
            (bs4_table, metadata) or None
        """
        metadata = {
            'source': None,
            'title_bs4': None,
            'is_extracted_title': False,
            'title_row_index': None,
        }
        
        logger.debug("\n=== TABLE EXTRACTION STRATEGIES ===")
        
        title_element = title_node.semantic_element
        title_text = normalize_text(title_element.text)
        
        logger.debug(f"Title text: '{title_text[:50]}...'")
        
        # Strategy 1: Check if title is in CompositeSemanticElement
        logger.debug("\n--- Strategy 1: CompositeSemanticElement sibling ---")
        parent_node = title_node.parent if hasattr(title_node, 'parent') else None
        
        if not parent_node:
            logger.debug("❌ No parent node found")
        elif not isinstance(parent_node.semantic_element, CompositeSemanticElement):
            logger.debug(f"❌ Parent is {type(parent_node.semantic_element).__name__}, not CompositeSemanticElement")
        else:
            logger.debug("✓ Title is in CompositeSemanticElement - checking siblings")
            
            siblings_found = 0
            for sibling_node in parent_node.children:
                siblings_found += 1
                sibling_type = type(sibling_node.semantic_element).__name__
                logger.debug(f"  Sibling #{siblings_found}: {sibling_type}")
                
                if isinstance(sibling_node.semantic_element, TableElement):
                    # Get table HTML
                    table_html = sibling_node.semantic_element.html_tag.get_source_code()
                    table_bs4 = BeautifulSoup(table_html, 'html.parser').find('table')
                    
                    # For metadata, we need title_bs4 from Strategy 1
                    title_html = title_element.html_tag.get_source_code()
                    title_bs4 = BeautifulSoup(title_html, 'html.parser').find()
                    metadata['title_bs4'] = title_bs4
                    
                    logger.debug(f"  Found TableElement sibling, checking if data table...")
                    
                    if not self._is_title_wrapper_table(table_bs4, title_bs4):
                        logger.info("✓✓ STRATEGY 1 SUCCESS: Found data table via CompositeSemanticElement")
                        metadata['source'] = 'composite'
                        metadata['is_extracted_title'] = True
                        return table_bs4, metadata
                    else:
                        logger.debug("  TableElement is just a wrapper, continuing search")
            
            logger.debug(f"❌ Strategy 1 failed: No suitable TableElement among {siblings_found} siblings")
        
        # Strategy 2: Check if title itself is inside a table
        logger.debug("\n--- Strategy 2: Title inside table ---")
        
        # Get title as bs4 for Strategy 2
        title_html = title_element.html_tag.get_source_code()
        title_bs4_isolated = BeautifulSoup(title_html, 'html.parser').find()
        
        # IMPORTANT: For Strategy 2, we need to find the title in the FULL soup
        # because parent_table lookup won't work on isolated element
        if full_soup:
            # Find title in full soup by text matching
            title_bs4 = self._find_element_in_soup(full_soup, title_text)
            if not title_bs4:
                logger.debug("❌ Could not locate title in full document soup")
                title_bs4 = title_bs4_isolated  # Fallback to isolated
        else:
            logger.debug("⚠️  No full_soup provided, using isolated title element")
            title_bs4 = title_bs4_isolated
        
        metadata['title_bs4'] = title_bs4
        logger.debug(f"Title HTML tag: <{title_bs4.name}>")
        
        parent_table = title_bs4.find_parent('table')
        
        if not parent_table:
            logger.debug("❌ Title is not inside any <table>")
        else:
            logger.debug(f"✓ Title is inside a <table> with {len(parent_table.find_all('tr'))} rows")
            
            if not self._is_title_wrapper_table(parent_table, title_bs4):
                logger.info("✓✓ STRATEGY 2 SUCCESS: Title is inside a data table")
                
                title_row = title_bs4.find_parent('tr')
                if title_row:
                    all_rows = parent_table.find_all('tr')
                    try:
                        title_row_index = all_rows.index(title_row)
                        metadata['title_row_index'] = title_row_index
                        logger.debug(f"Title is in row {title_row_index}")
                    except ValueError:
                        logger.warning("Could not determine title row index")
                
                metadata['source'] = 'parent_table'
                return parent_table, metadata
            else:
                logger.debug("❌ Parent table is just a wrapper")
        
        # Strategy 3: Look for next table after title
        logger.debug("\n--- Strategy 3: Next table after title ---")
        logger.info("Looking for next table after title element")
        
        # For Strategy 3, we MUST have title in full soup context
        if not full_soup:
            logger.error("❌ Cannot use Strategy 3: No full_soup provided!")
            logger.error("❌❌ ALL STRATEGIES FAILED: Could not find table for title")
            logger.error("=== END TABLE EXTRACTION ===\n")
            return None
        
        # Find title element in full soup
        title_bs4_in_context = self._find_element_in_soup(full_soup, title_text)
        
        if not title_bs4_in_context:
            logger.error("❌ Could not locate title in full document soup!")
            logger.error("This means the title text doesn't appear in the document HTML,")
            logger.error("which suggests sec-parser modified or extracted it.")
            html_context = dump_html_context(title_bs4, context_lines=5)
            logger.debug(html_context)
            return None
        
        logger.debug(f"✓ Found title in full document at <{title_bs4_in_context.name}>")
        
        next_table = self._find_next_table(title_bs4_in_context)
        
        if next_table:
            logger.info("✓✓ STRATEGY 3 SUCCESS: Found next table after title")
            metadata['source'] = 'next_table'
            return next_table, metadata
        
        # ALL STRATEGIES FAILED - dump diagnostic info
        logger.error("❌❌ ALL STRATEGIES FAILED: Could not find table for title")
        logger.error("=== END TABLE EXTRACTION ===\n")
        
        # Dump HTML context for debugging
        html_context = dump_html_context(title_bs4_in_context, context_lines=5)
        logger.debug(html_context)
        
        return None
    
    def _find_element_in_soup(
        self, 
        soup: BeautifulSoup, 
        target_text: str
    ) -> Optional[Tag]:
        """
        Find element in soup by text content.
        
        This is critical for Strategy 2 and 3 to work, because we need
        to find the title element in the FULL document soup, not the
        isolated HTML from get_source_code().
        
        Args:
            soup: Full document BeautifulSoup
            target_text: Text to search for
            
        Returns:
            Tag element or None
        """
        normalized_target = normalize_text(target_text)
        spaceless_target = normalized_target.replace(" ", "")
        
        if not spaceless_target:
            return None
        
        # Try to find exact text match
        for element in soup.find_all(['div', 'span', 'p', 'td', 'th', 'b', 'strong', 'i', 'em']):
            # Skip elements inside tables (we want the title BEFORE the table)
            if element.find_parent('table'):
                continue
            
            element_text = normalize_text(element.get_text("", strip=True))
            spaceless_element_text = element_text.replace(" ", "")
            
            # Exact match
            if spaceless_element_text == spaceless_target:
                logger.debug(f"Found exact text match in <{element.name}> tag")
                return element
            
            # Starts with match (title might be truncated)
            if spaceless_element_text.startswith(spaceless_target) and len(spaceless_element_text) < len(spaceless_target) * 1.5:
                logger.debug(f"Found 'starts with' match in <{element.name}> tag")
                return element
        
        logger.warning(f"Could not find element with text: '{target_text[:50]}...'")
        return None
    
    def _find_next_table(self, element: Tag) -> Optional[Tag]:
        """Find next data table after given element."""
        if not element:
            return None
        
        current_table = element.find_next('table')
        table_counter = 0
        rejection_reasons = []
        
        while current_table:
            table_counter += 1
            rows = current_table.find_all('tr')
            num_rows = len(rows)
            
            # Get table preview
            table_text_preview = current_table.get_text(strip=True)[:100]
            logger.debug(f"\n--- Examining Table #{table_counter} ---")
            logger.debug(f"Rows: {num_rows}")
            logger.debug(f"Preview: {table_text_preview}...")
            
            # Check 1: Minimum row count
            if num_rows < 3:
                reason = f"Table #{table_counter}: Too small ({num_rows} rows, need ≥3)"
                logger.debug(f"REJECTED: {reason}")
                rejection_reasons.append(reason)
                current_table = current_table.find_next('table')
                continue
            
            # Check 2: Title wrapper detection
            is_wrapper = self._is_title_wrapper_table(current_table)
            if is_wrapper:
                # Get more details about why it's a wrapper
                cells_with_content = sum(
                    1 for row in current_table.find_all('tr')
                    for cell in row.find_all(['td', 'th'])
                    if cell.get_text(strip=True)
                )
                reason = f"Table #{table_counter}: Detected as title wrapper ({cells_with_content} cells with content)"
                logger.debug(f"REJECTED: {reason}")
                rejection_reasons.append(reason)
                current_table = current_table.find_next('table')
                continue
            
            # Check 3: Look for data characteristics
            has_numbers = any(
                any(c.isdigit() for c in cell.get_text())
                for row in rows[:10]  # Check first 10 rows
                for cell in row.find_all(['td', 'th'])
            )
            
            if not has_numbers:
                reason = f"Table #{table_counter}: No numeric data found in first 10 rows"
                logger.warning(f"SUSPICIOUS: {reason} (but accepting anyway)")
            
            # ACCEPTED
            logger.info(f"✓ ACCEPTED Table #{table_counter}: {num_rows} rows, has_numbers={has_numbers}")
            logger.debug(f"Total tables examined: {table_counter}, rejected: {len(rejection_reasons)}")
            return current_table
        
        # No table found - log comprehensive summary
        logger.warning(f"No suitable table found after element")
        logger.warning(f"Summary: Examined {table_counter} tables, all rejected:")
        for reason in rejection_reasons:
            logger.warning(f"  - {reason}")
        
        if table_counter == 0:
            logger.error("CRITICAL: No <table> tags found after title element at all!")
            logger.error("This suggests the title and table might be in different sections")
        
        return None
    
    def _is_title_wrapper_table(
        self,
        table: Tag,
        title_element: Optional[Tag] = None
    ) -> bool:
        """
        Detect if a table is just a styling wrapper for a title.
        
        Args:
            table: Table to check
            title_element: Optional title element (if inside table, NOT a wrapper)
            
        Returns:
            True if wrapper, False if data table
        """
        # Rule 1: If title is a cell in this table, NOT a wrapper
        if title_element and title_element.name in ['td', 'th']:
            if title_element.find_parent('table') == table:
                logger.debug("  Wrapper check: Title is a cell inside table → NOT a wrapper")
                return False
        
        rows = table.find_all('tr')
        num_rows = len(rows)
        
        # Rule 2: Title wrappers usually have few rows
        if num_rows > 5:
            logger.debug(f"  Wrapper check: Too many rows ({num_rows}) → NOT a wrapper")
            return False
        
        # Count cells with actual content
        cells_with_content = 0
        total_text = []
        total_cells = 0
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            total_cells += len(cells)
            for cell in cells:
                text = cell.get_text(strip=True)
                if text:
                    cells_with_content += 1
                    total_text.append(text)
        
        # Rule 3: Very few cells with content = likely wrapper
        if cells_with_content <= 2:
            combined_text = ' '.join(total_text)
            logger.debug(f"  Wrapper check: Only {cells_with_content} cells with content → IS a wrapper")
            logger.debug(f"  Wrapper text: '{combined_text[:50]}...'")
            return True
        
        # Additional diagnostic info
        content_ratio = cells_with_content / total_cells if total_cells > 0 else 0
        logger.debug(f"  Wrapper check: {cells_with_content}/{total_cells} cells have content "
                    f"({content_ratio:.1%}) → NOT a wrapper")
        
        return False