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
    
    def find_statement_titles_in_elements(
        self,
        elements: list,
        keywords: StatementKeywords
    ) -> List[Dict]:
        """Find TitleElements matching statement keywords in a flat list."""
        from sec_parser.semantic_elements.title_element import TitleElement
        
        matching_titles = []
        
        for element in elements:
            if isinstance(element, TitleElement):
                title_text = normalize_text(element.text)
                title_lower = title_text.lower()
                
                # Must have at least one required keyword
                has_required = any(kw in title_lower for kw in keywords.must_have)
                if not has_required:
                    continue
                
                logger.info(f"Found matching title: '{title_text}'")
                matching_titles.append({
                    'text': title_text,
                    'element': element
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
    
    def _find_element_in_soup(
        self, 
        soup: BeautifulSoup, 
        title_element_html_tag,
        target_text: str
    ) -> Optional[Tag]:
        """
        Find the actual title element in the full document by extracting
        the innermost content element and searching by its unique attributes.
        
        When TableTitleSplitter runs, it wraps title rows in new table structures.
        We need to find the actual content element (usually a <span>) inside.
        
        Args:
            soup: Full document BeautifulSoup
            title_element_html_tag: The HtmlTag object from sec-parser
            target_text: The text content
            
        Returns:
            Tag element or None
        """
        # Get the HTML source
        title_html = title_element_html_tag.get_source_code()
        
        # Parse it to extract attributes
        temp_soup = BeautifulSoup(title_html, 'html.parser')
        
        # The structure might be: <table><tr><td><span>Title</span></td></tr></table>
        # We need to find the innermost element that contains the actual text
        
        # Find the deepest element that contains the full text
        innermost_element = None
        normalized_target_text = normalize_text(target_text)
        
        for element in temp_soup.find_all():
            element_text = normalize_text(element.get_text("", strip=True))
            if element_text == normalized_target_text:
                # Check if this element has no children with text, or only has the text directly
                if not any(child for child in element.find_all() 
                        if normalize_text(child.get_text("", strip=True)) == normalized_target_text):
                    innermost_element = element
                    break
        
        if not innermost_element:
            logger.warning("Could not find innermost element in title HTML")
            return None
        
        # Now we have the innermost element (e.g., the <span>)
        # Search for it in the full document by matching attributes
        tag_name = innermost_element.name
        
        # Build a set of attributes to match
        attrs_to_match = {}
        if innermost_element.get('style'):
            attrs_to_match['style'] = innermost_element.get('style')
        if innermost_element.get('class'):
            attrs_to_match['class'] = innermost_element.get('class')
        
        logger.debug(f"Searching for <{tag_name}> with attributes: {attrs_to_match}")
        
        # Search for matching elements in the full document
        candidates = []
        
        for element in soup.find_all(tag_name):
            # Check if attributes match
            attrs_match = True
            
            if 'style' in attrs_to_match:
                if element.get('style') != attrs_to_match['style']:
                    attrs_match = False
            
            if 'class' in attrs_to_match:
                if element.get('class') != attrs_to_match['class']:
                    attrs_match = False
            
            if not attrs_match:
                continue
            
            # Check if text matches
            element_text = normalize_text(element.get_text("", strip=True))
            if element_text == normalized_target_text:
                candidates.append(element)
        
        if not candidates:
            logger.warning(f"Could not find any <{tag_name}> elements matching attributes and text")
            return None
        
        if len(candidates) > 1:
            logger.warning(f"Found {len(candidates)} matching elements - using the first one")
            # Prefer elements NOT inside tables (for Strategy 3)
            # But allow elements inside tables (for Strategy 2)
            for candidate in candidates:
                if not candidate.find_parent('table'):
                    logger.debug("Selected candidate outside of table")
                    return candidate
            
            logger.debug("All candidates are inside tables - using first")
        
        logger.debug(f"Found matching element: <{candidates[0].name}>")
        return candidates[0]


    def _extract_table_from_semantic_node(
        self, 
        title_node,
        full_soup: Optional[BeautifulSoup] = None
    ) -> Optional[Tuple[Tag, Dict]]:
        """
        Extract table using the title element's attributes to locate it in the document.
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
                    table_html = sibling_node.semantic_element.html_tag.get_source_code()
                    table_bs4 = BeautifulSoup(table_html, 'html.parser').find('table')
                    
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
        
        # For Strategy 2 and 3, find the title in the full document
        if not full_soup:
            logger.error("❌ Cannot use Strategy 2/3: No full_soup provided!")
            return None
        
        # Find the title element by its HTML attributes
        title_bs4 = self._find_element_in_soup(full_soup, title_element.html_tag, title_text)
        
        if not title_bs4:
            logger.error("❌ Could not locate title in full document!")
            return None
        
        metadata['title_bs4'] = title_bs4
        logger.debug(f"✓ Found title in full document: <{title_bs4.name}>")
        
        # Strategy 2: Check if title is inside a table
        logger.debug("\n--- Strategy 2: Title inside table ---")
        
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
        
        next_table = self._find_next_table(title_bs4)
        
        if next_table:
            logger.info("✓✓ STRATEGY 3 SUCCESS: Found next table after title")
            metadata['source'] = 'next_table'
            return next_table, metadata
        
        logger.error("❌❌ ALL STRATEGIES FAILED")
        logger.error("=== END TABLE EXTRACTION ===\n")
        
        html_context = dump_html_context(title_bs4, context_lines=5)
        logger.debug(html_context)
        
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