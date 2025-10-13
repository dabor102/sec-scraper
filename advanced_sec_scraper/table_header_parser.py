"""
Table header parsing with multiple strategies for different layouts.
"""

import logging
import re
from datetime import datetime
from typing import List, Dict, Optional, Set
from dateutil.parser import parse as parse_date
from dateutil.parser._parser import ParserError

from .config import Patterns
from .table_processor import normalize_text

logger = logging.getLogger(__name__)


# ============================================================================
# PERIOD PARSING
# ============================================================================

class TableHeaderParser:
    """Parses table headers to extract financial periods."""
    
    def __init__(self, fiscal_calendar: Dict[str, str]):
        """
        Initialize parser with fiscal calendar.
        
        Args:
            fiscal_calendar: Mapping of month names to quarters (Q1-Q4)
        """
        self.fiscal_calendar = fiscal_calendar
    
    def parse_header(
        self,
        header_rows: List[List[str]],
        metric_col: int,
        is_snapshot: bool = False
    ) -> List[str]:
        """
        Parse header rows to extract periods.
        Tries multiple strategies in order of complexity.
        
        Args:
            header_rows: List of header row data
            metric_col: Column index for metrics
            is_snapshot: Whether this is a balance sheet (point-in-time)
            
        Returns:
            List of period strings (e.g., ["Q3 2024", "Q2 2024"])
        """
        if not header_rows:
            return []
        
        # Strategy 1: Split header (description and years in separate rows)
        periods = self._try_split_header(header_rows, metric_col, is_snapshot)
        if periods:
            return periods
        
        # Strategy 2: Broadcasting (layered headers with shared context)
        periods = self._try_broadcasting_header(header_rows, metric_col, is_snapshot)
        if periods:
            return periods
        
        # Strategy 3: Horizontal layout (dates/years in same row)
        periods = self._try_horizontal_header(header_rows, metric_col, is_snapshot)
        if periods:
            return periods
        
        # Strategy 4: Vertical layout (simple column-by-column)
        periods = self._try_vertical_header(header_rows, metric_col, is_snapshot)
        return periods
    
    def parse_period(
        self,
        header_text: str,
        is_snapshot: bool = False
    ) -> Optional[str]:
        """
        Parse a single period string into standardized format.
        
        Args:
            header_text: Text to parse (e.g., "Three Months Ended September 30, 2024")
            is_snapshot: Whether this is a balance sheet
            
        Returns:
            Standardized period string or None
        """
        logger.debug(f"--- PARSING NEW PERIOD ---")
        logger.debug(f"Input header text: '{header_text}'")
        
        if not header_text or not header_text.strip():
            return None
        
        # Extract calendar year
        header_lower = header_text.lower()
        year_match = Patterns.YEAR.search(header_text)
        if not year_match:
            logger.debug(f"No year found in header.")
            return None
        
        calendar_year = int(year_match.group(1))
        logger.debug(f"Found calendar year: {calendar_year}")
        
        # Try to parse date and find quarter
        parsed_date, month_name, quarter = None, None, None
        
        # Only attempt fuzzy date parsing if a full date (including a day number) 
        # is explicitly present, typically recognizable by a number followed by a comma/space and the year.
        has_day = re.search(r'\d{1,2}[,\s]+\d{4}', header_text) is not None

        if has_day:
            try:
                parsed_date = parse_date(header_text, fuzzy=True)
                month_name = parsed_date.strftime('%B').lower()
                if month_name in self.fiscal_calendar:
                    quarter = self.fiscal_calendar[month_name]
                logger.debug(f"Date parsing successful: date='{parsed_date.date()}', "
                            f"month='{month_name}', mapped_quarter='{quarter}'")
            except (ValueError, OverflowError, AttributeError):
                logger.debug(f"Date parsing failed. Falling back to keyword search.")
        
        if not quarter:
            # Fallback: search for month names
            for month, q in self.fiscal_calendar.items():
                if month in header_lower:
                    month_name, quarter = month, q
                    logger.debug(f"Fallback found month='{month_name}', "
                               f"mapped_quarter='{quarter}'")
                    break
        
        if not quarter:
            # Check for explicit quarter names (First, Second, Third, Fourth Quarter)
            if re.search(r'\bfirst\s+quarter\b', header_lower):
                quarter = 'Q1'
                logger.debug("Explicit quarter keyword found: Q1")
            elif re.search(r'\bsecond\s+quarter\b', header_lower):
                quarter = 'Q2'
                logger.debug("Explicit quarter keyword found: Q2")
            elif re.search(r'\bthird\s+quarter\b', header_lower):
                quarter = 'Q3'
                logger.debug("Explicit quarter keyword found: Q3")
            elif re.search(r'\bfourth\s+quarter\b', header_lower):
                quarter = 'Q4'
                logger.debug("Explicit quarter keyword found: Q4")

        
        if not quarter:
            logger.debug(f"Could not determine quarter from text.")
            if is_snapshot or Patterns.QUARTER.search(header_lower):
                return None
        
        # Determine fiscal year
        logger.debug(f"Determining fiscal year for calendar_year={calendar_year}, "
                    f"month='{month_name}'")
        fiscal_year = self._determine_fiscal_year(calendar_year, month_name, parsed_date)
        logger.debug(f"Final fiscal year: {fiscal_year}")
        
        # Construct period string
        final_period = None
        if is_snapshot:
            final_period = f"{quarter} {fiscal_year}"
        elif Patterns.FY.search(header_lower):
            final_period = f"FY {fiscal_year}"
        elif Patterns.NINE_MONTHS.search(header_lower):
            final_period = f"Nine Months {fiscal_year}"
        elif Patterns.SIX_MONTHS.search(header_lower) or re.search(r'\b(first|second|third|fourth)\s+six\s+months\b', header_lower):
            final_period = f"Six Months {fiscal_year}"
        elif Patterns.QUARTER.search(header_lower) or re.search(r'\b(first|second|third|fourth)\s+quarter\b', header_lower):
            final_period = f"{quarter} {fiscal_year}"
        elif quarter:
            final_period = f"{quarter} {fiscal_year}"
        
        if final_period:
            logger.debug(f"SUCCESS: Final period string is '{final_period}'")
        else:
            logger.warning(f"FAILURE: Could not classify period for '{header_text}'")
        
        return final_period
    
    def _determine_fiscal_year(
        self,
        calendar_year: int,
        month_name: str,
        parsed_date
    ) -> int:
        """
        Determine correct fiscal year for a given date.
        
        The fiscal year is the year in which the fiscal period ENDS.
        For example, if FYE is January 31, a balance sheet dated Jan 31, 2025
        represents the end of fiscal year 2024.
        """
        # Find FYE month
        try:
            q4_months = [m for m, q in self.fiscal_calendar.items() if q == 'Q4']
            if not q4_months:
                logger.warning("Could not determine Q4 months. Returning calendar year.")
                return calendar_year
            fye_month_name = q4_months[-1]
            fye_month = datetime.strptime(fye_month_name, '%B').month
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to determine FYE month: {e}. Returning calendar year.")
            return calendar_year
        
        # Determine month of statement date
        if parsed_date:
            date_month = parsed_date.month
        else:
            if month_name is None:
                logger.warning("Could not determine month name. Returning calendar year.")
                return calendar_year
            
            try:
                date_month = datetime.strptime(month_name, '%B').month
            except ValueError:
                logger.warning(f"Could not parse month name '{month_name}'. "
                             "Returning calendar year.")
                return calendar_year
        
        # Fiscal year logic
        fy_start_month = (fye_month % 12) + 1
        
        if date_month < fy_start_month:
            fiscal_year = calendar_year - 1
            logger.debug(f"Date month {date_month} is before FY start {fy_start_month}. "
                        f"Fiscal Year: {calendar_year} -> {fiscal_year}")
        else:
            fiscal_year = calendar_year
            logger.debug(f"Date month {date_month} is on/after FY start {fy_start_month}. "
                        f"Fiscal Year: {calendar_year}")
        
        return fiscal_year
    
    # ========================================================================
    # HEADER PARSING STRATEGIES
    # ========================================================================
    def _try_split_header(
        self,
        header_rows: List[List[str]],
        metric_col: int,
        is_snapshot: bool
    ) -> List[str]:
        """Handle split headers (description/month/year in separate rows)."""
        logger.debug("Attempting improved split header format")
        
        description_patterns = [
            re.compile(r'(?:fiscal\s+)?(?:years?|months?|quarters?)\s*ended', re.IGNORECASE),
            re.compile(r'(?:three|six|nine|twelve)\s*months?\s*ended', re.IGNORECASE),
            re.compile(r'as\s+of', re.IGNORECASE),
        ]
        
        for desc_row_idx in range(len(header_rows) - 1):
            desc_row = header_rows[desc_row_idx]
            
            # Find description cells
            desc_map = {}
            for col_idx, cell in enumerate(desc_row):
                cell_text = cell.strip()
                if (col_idx > metric_col and cell_text and 
                    any(p.search(cell_text) for p in description_patterns)):
                    desc_map[col_idx] = cell_text
            
            if not desc_map:
                continue
            
            logger.debug(f"Found description row {desc_row_idx} with descriptions: {desc_map}")
            
            # Look for year rows
            for year_row_idx in range(desc_row_idx + 1, min(desc_row_idx + 4, len(header_rows))):
                year_row = header_rows[year_row_idx]
                
                year_map = {}
                for col_idx, cell in enumerate(year_row):
                    if col_idx > metric_col and cell.strip() and Patterns.YEAR.search(cell):
                        year_map[col_idx] = cell.strip()
                
                if not year_map:
                    continue
                
                logger.debug(f"Found year row {year_row_idx} with years: {year_map}")
                
                # Build month map from intermediate rows (NEW)
                month_map = {}
                if year_row_idx > desc_row_idx + 1:
                    for month_row_idx in range(desc_row_idx + 1, year_row_idx):
                        month_row = header_rows[month_row_idx]
                        for col_idx, cell in enumerate(month_row):
                            cell_text = cell.strip()
                            if col_idx > metric_col and cell_text:
                                # Store non-empty cells that might contain dates/months
                                month_map[col_idx] = cell_text
                                logger.debug(f"Found month/date at col {col_idx}: '{cell_text}'")
                
                periods = []
                parsed_combinations = set()
                sorted_year_cols = sorted(year_map.keys())
                
                for year_col in sorted_year_cols:
                    year_text = year_map[year_col]
                    
                    # Find closest description
                    if not desc_map:
                        continue

                    closest_desc_col = min(
                        desc_map.keys(), 
                        key=lambda desc_c: abs(desc_c - year_col)
                    )
                    best_desc = desc_map[closest_desc_col]
                    
                    # Find closest month text (NEW - same logic as description)
                    month_text = ''
                    if month_map:
                        closest_month_col = min(
                            month_map.keys(),
                            key=lambda month_c: abs(month_c - year_col)
                        )
                        # Only use month if it's reasonably close (within same logical group)
                        if abs(closest_month_col - year_col) <= 6:
                            month_text = month_map[closest_month_col]
                            logger.debug(f"Matched year at col {year_col} with month at col {closest_month_col}: '{month_text}'")
                    
                    if best_desc:
                        combined_text = f"{best_desc} {month_text} {year_text}"
                        period = self.parse_period(combined_text, is_snapshot)
                        
                        combination_key = (period, year_text)

                        if period and combination_key not in parsed_combinations:
                            periods.append(period)
                            parsed_combinations.add(combination_key)
                            logger.debug(f"Matched year at col {year_col} ('{year_text}') "
                                    f"with closest desc at col {closest_desc_col} -> '{period}'")
                    else:
                        logger.warning(f"Could not find matching description for year at col {year_col}")
                
                if periods:
                    logger.info(f"Successfully parsed split header: {periods}")
                    unique_periods = list(dict.fromkeys(periods))
                    return unique_periods
        
        return []

    def _try_horizontal_header(
        self,
        header_rows: List[List[str]],
        metric_col: int,
        is_snapshot: bool
    ) -> List[str]:
        """Parse horizontal header (description on left, dates/years on right)."""
        logger.debug("Attempting horizontal header format")
        
        for row in header_rows:
            if not any(cell.strip() for cell in row):
                continue
            
            base_desc_cells = [cell.strip() for cell in row[:metric_col + 1] if cell.strip()]
            data_cells = row[metric_col + 1:]
            
            # Need description and cells with years
            year_containing_cells = [cell for cell in data_cells if Patterns.YEAR.search(cell)]
            
            if not base_desc_cells or not year_containing_cells:
                continue
            
            base_description = normalize_text(' '.join(base_desc_cells))
            periods = []
            
            for cell in year_containing_cells:
                cell_stripped = cell.strip()
                if not cell_stripped:
                    continue
                
                combined_text = f"{base_description} {cell_stripped}"
                period = self.parse_period(combined_text, is_snapshot)
                if period and period not in periods:
                    periods.append(period)
            
            if periods:
                logger.info(f"Successfully parsed horizontal header: {periods}")
                return periods
        
        return []
    
    def _try_broadcasting_header(
        self,
        header_rows: List[List[str]],
        metric_col: int,
        is_snapshot: bool
    ) -> List[str]:
        """
        Broadcasting approach for layered headers.
        """
        logger.debug("Attempting broadcasting header format")
        
        # 1. Capture shared descriptions from metric column (master context)
        shared_row_descriptions = []
        for row in header_rows:
            if len(row) > metric_col:
                cell_text = normalize_text(row[metric_col])
                classifications = self._classify_cell_text(cell_text)
                if 'DURATION' in classifications or 'DATE' in classifications:
                    shared_row_descriptions.append(cell_text)
                    logger.debug(f"Found shared row description: '{cell_text}'")
        
        master_description = normalize_text(' '.join(shared_row_descriptions))
        
        # 2. Process all cells in data columns (right of metric_col)
        processed_cells = []
        for row_idx, row in enumerate(header_rows):
            cells = row[metric_col + 1:]
            for col_pos_relative, cell in enumerate(cells):
                cell_text = cell.strip()
                if cell_text:
                    processed_cells.append({
                        'text': cell_text,
                        'position': col_pos_relative,
                        'classifications': self._classify_cell_text(cell_text),
                        'row': row_idx
                    })
        
        if not processed_cells:
            return []
        
        # 3. Determine the base row (row with most periods) and map positions
        base_row_cells = []
        max_len = 0
        
        # Find the row that contributes the maximum number of periods
        for row_idx in range(len(header_rows)):
            current_row_cells = [c for c in processed_cells if c['row'] == row_idx]
            if len(current_row_cells) > max_len:
                max_len = len(current_row_cells)
                base_row_cells = current_row_cells

        if not base_row_cells:
            return []

        num_periods = len(base_row_cells)
        logger.debug(f"Broadcasting: detected {num_periods} periods from base row {base_row_cells[0]['row']}")
        
        # Map period index (0 to N-1) to its relative column position
        position_to_column = {i: cell['position'] for i, cell in enumerate(base_row_cells)}
        resolved_contexts = [{'duration': [], 'date': [], 'year': []} for _ in range(num_periods)]
        
        # 4. Broadcast information from all cells onto the base periods
        
        # Separate spanning context cells from non-spanning cells
        spanning_context_cells = []
        non_spanning_cells = []
        for cell_info in processed_cells:
            is_spanning_context = 'DURATION' in cell_info['classifications'] or 'DATE' in cell_info['classifications']
            # Only consider cells not in the base row as spanning context to avoid self-application issues
            base_row_idx = base_row_cells[0]['row']
            if is_spanning_context and cell_info['row'] != base_row_idx:
                 spanning_context_cells.append(cell_info)
            else:
                non_spanning_cells.append(cell_info)

        # Apply NON-spanning context (mostly base year row)
        for cell_info in non_spanning_cells:
            col_pos_relative = cell_info['position']
            best_period_idx = min(
                range(num_periods),
                key=lambda i: abs(col_pos_relative - position_to_column[i])
            )
            closest_dist = abs(col_pos_relative - position_to_column[best_period_idx])
            
            if closest_dist <= 1:
                idx_to_update = best_period_idx
                if 'DURATION' in cell_info['classifications']:
                    resolved_contexts[idx_to_update]['duration'].append(cell_info['text'])
                if 'DATE' in cell_info['classifications']:
                    resolved_contexts[idx_to_update]['date'].append(cell_info['text'])
                if 'YEAR' in cell_info['classifications']:
                    resolved_contexts[idx_to_update]['year'].append(cell_info['text'])
        
        # Apply SPANNING context (e.g., "Second Quarter")
        # Find start columns for each period group (P0, P2, P4, etc.)
        group_starts = [position_to_column[i] for i in range(0, num_periods, 2)]
        group_map = {} # Map a starting column position to a list of periods it covers
        for i in range(0, num_periods, 2):
            if i < num_periods:
                start_col = position_to_column[i]
                group_map[start_col] = [i]
                if i + 1 < num_periods:
                    group_map[start_col].append(i + 1)

        for cell_info in spanning_context_cells:
            col_pos_relative = cell_info['position']
            text = cell_info['text']
            classifications = cell_info['classifications']
            
            # Find the closest *group start* (P0 or P2)
            if not group_starts:
                continue

            closest_group_start = min(
                group_starts,
                key=lambda start_col: abs(col_pos_relative - start_col)
            )

            # Check if the spanning text is close to the start of its intended group (P0 or P2)
            # Use a generous threshold (e.g., <= 3 columns) for spanning cells to account for padding
            if abs(col_pos_relative - closest_group_start) <= 3: 
                periods_to_update = group_map.get(closest_group_start, [])
                
                for period_idx in periods_to_update:
                    if 'DURATION' in classifications:
                        resolved_contexts[period_idx]['duration'].append(text)
                    if 'DATE' in classifications:
                        resolved_contexts[period_idx]['date'].append(text)
                    if 'YEAR' in classifications:
                        resolved_contexts[period_idx]['year'].append(text)


        # 5. Reconstruct and parse periods
        ordered_periods = []
        for idx, ctx in enumerate(resolved_contexts):
            duration_text = normalize_text(' '.join(dict.fromkeys(ctx['duration'])))
            date_text = normalize_text(' '.join(dict.fromkeys(ctx['date'])))
            year_text = normalize_text(' '.join(dict.fromkeys(ctx['year'])))
            
            # Ensure the base period's text (usually the year) is included
            base_cell_text = base_row_cells[idx]['text']
            if not year_text and 'YEAR' in base_row_cells[idx]['classifications']:
                 year_text = base_cell_text
            
            combined_text = f"{master_description} {duration_text} {date_text} {year_text}".strip()
            period = self.parse_period(combined_text, is_snapshot)
            
            if period:
                ordered_periods.append(period)
                logger.debug(f"Period {idx}: '{combined_text}' -> {period}")
            else:
                logger.warning(f"Period {idx}: Failed to parse -> '{combined_text}'")

        
        # 6. Final return
        if ordered_periods and len(ordered_periods) == num_periods:
            logger.info(f"Successfully parsed {len(ordered_periods)} periods via broadcasting")
            # Remove duplicates while preserving order
            unique_periods = list(dict.fromkeys(ordered_periods))
            return unique_periods
        
        return []

    def _try_vertical_header(
        self,
        header_rows: List[List[str]],
        metric_col: int,
        is_snapshot: bool
    ) -> List[str]:
        """Simple vertical header parsing."""
        if not header_rows:
            return []
        
        max_cols = max(len(row) for row in header_rows) if header_rows else 0
        if max_cols <= metric_col + 1:
            return []
        
        # Build column content map
        column_content = {}
        for row in header_rows:
            for col_idx in range(metric_col + 1, len(row)):
                cell_text = row[col_idx].strip()
                if cell_text:
                    if col_idx not in column_content:
                        column_content[col_idx] = []
                    if not column_content[col_idx] or column_content[col_idx][-1] != cell_text:
                        column_content[col_idx].append(cell_text)
        
        if not column_content:
            return []
        
        # Group nearby columns
        period_groups = []
        sorted_cols = sorted(column_content.keys())
        current_group = []
        last_col = -1
        
        for col_idx in sorted_cols:
            if last_col != -1 and col_idx - last_col > 3:
                if current_group:
                    period_groups.append(current_group)
                current_group = []
            current_group.append(col_idx)
            last_col = col_idx
        
        if current_group:
            period_groups.append(current_group)
        
        # Parse periods
        periods = []
        for group in period_groups:
            group_texts = []
            for col_idx in group:
                group_texts.extend(column_content[col_idx])
            
            unique_texts = list(dict.fromkeys(group_texts))
            combined_text = ' '.join(unique_texts)
            
            period = self.parse_period(combined_text, is_snapshot)
            if period and (not periods or period != periods[-1]):
                periods.append(period)
                logger.debug(f"Parsed period from columns {group}: "
                           f"'{combined_text}' -> {period}")
        
        return periods
    
    def _classify_cell_text(self, text: str) -> Set[str]:
        """Analyze table cell text and return information types it contains."""
        if not text or not text.strip():
            return set()
        
        classifications = set()
        text_lower = text.lower()
        text_cleaned = text.strip()
        
        # Check for duration keywords - EXPANDED to include standalone period phrases
        duration_keywords = [
            'months ended', 
            'year ended', 
            'fiscal year', 
            'quarter',
            'six months',      # Added
            'nine months',     # Added
            'twelve months',   # Added
            'three months',    # Added
            'first six months',   # Added
            'second quarter',     # Added
            'third quarter',      # Added
            'fourth quarter',     # Added
            'first quarter',      # Added
        ]
        if any(keyword in text_lower for keyword in duration_keywords):
            classifications.add('DURATION')
        
        # Check for standalone year
        if Patterns.YEAR_ONLY.match(text_cleaned):
            classifications.add('YEAR')
            return classifications
        
        # Try to parse date
        try:
            parse_date(text_cleaned, fuzzy=True)
            classifications.add('DATE')
            if Patterns.YEAR.search(text_cleaned):
                classifications.add('YEAR')
        except (ValueError, OverflowError):
            if Patterns.YEAR.search(text_cleaned):
                classifications.add('YEAR')
        
        return classifications

# ============================================================================
# PERIOD CORRECTION
# ============================================================================

def correct_fiscal_periods(periods: List[str]) -> List[str]:
    """
    Correct fiscal quarter periods based on cumulative context.
    Q1-Q3 implies Q3, Q1-Q2 implies Q2, standalone quarter implies Q1.
    """
    if not periods:
        return []
    
    year_anchors = {}
    single_quarter_pattern = re.compile(r'^(Q\d) (\d{4})$')
    
    # Find cumulative period anchors
    for period in periods:
        if 'Q1-Q3' in period:
            year = period.split()[-1]
            year_anchors[year] = 'Q3'
        elif 'Q1-Q2' in period:
            year = period.split()[-1]
            year_anchors[year] = 'Q2'
    
    # If no anchors and only one single-quarter period, assume Q1
    if not year_anchors:
        single_quarters = [p for p in periods if single_quarter_pattern.match(p)]
        if len(single_quarters) == 1:
            logger.info("Only one single-quarter period found. Assuming it's Fiscal Q1.")
            q1_period = single_quarters[0]
            year = q1_period.split()[-1]
            return [f'Q1 {year}' if p == q1_period else p for p in periods]
        return periods
    
    # Apply corrections
    corrected_periods = []
    for period in periods:
        match = single_quarter_pattern.match(period)
        if match:
            year = match.group(2)
            if year in year_anchors:
                corrected_period = f"{year_anchors[year]} {year}"
                logger.info(f"Correcting '{period}' to '{corrected_period}' "
                          "based on cumulative anchor.")
                corrected_periods.append(corrected_period)
            else:
                corrected_periods.append(period)
        else:
            corrected_periods.append(period)
    
    return corrected_periods