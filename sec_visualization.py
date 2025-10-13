import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SECDataProcessor:
    """Process SEC filing data from CSV files"""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        
    def normalize_units(self, value: float, units: str) -> float:
        """Normalize all values to millions"""
        if pd.isna(value) or pd.isna(units):
            return value
            
        units_lower = str(units).lower().strip()
        
        # Convert to millions
        if 'thousand' in units_lower:
            return value / 1000
        elif 'billion' in units_lower:
            return value * 1000
        elif 'million' in units_lower:
            return value
        else:
            # Assume base units, convert to millions
            return value / 1_000_000
    
    def load_csv_files(self, file_paths: List[str]) -> pd.DataFrame:
        """Load multiple CSV files and combine them"""
        dfs = []
        
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                # Drop URL column if it exists
                if 'URL' in df.columns:
                    df = df.drop('URL', axis=1)
                dfs.append(df)
                logger.info(f"Loaded {file_path}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
        if not dfs:
            return pd.DataFrame()
            
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    
    def extract_company_info(self, filename: str) -> Tuple[str, str, str]:
        """Extract ticker, filing type, and quarter range from filename"""
        # Pattern: TICKER_FILING-TYPE_QUARTERS.csv
        pattern = r'([A-Z]+)_(10-[QK])_(.+)\.csv'
        match = re.search(pattern, filename)
        
        if match:
            return match.group(1), match.group(2), match.group(3)
        return None, None, None
    
    def fill_missing_sections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing Section values with Metric values"""
        df = df.copy()
        mask = df['Section'].isna() | (df['Section'] == '')
        df.loc[mask, 'Section'] = df.loc[mask, 'Metric']
        logger.info(f"Filled {mask.sum()} missing Section values with Metric names")
        return df
    
    def normalize_all_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize all values to millions"""
        df = df.copy()
        df['Value_Millions'] = df.apply(
            lambda row: self.normalize_units(row['Value'], row['Units']),
            axis=1
        )
        df['Units_Normalized'] = 'Millions'
        logger.info("Normalized all values to millions")
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate entries"""
        initial_count = len(df)
        
        # Define key columns for identifying duplicates
        key_columns = [
            'Filing Date', 'Report Period End Date', 'Financial Section',
            'Section', 'Metric', 'Period'
        ]
        
        df = df.drop_duplicates(subset=key_columns, keep='first')
        removed = initial_count - len(df)
        logger.info(f"Removed {removed} duplicate rows")
        return df
    
    def calculate_q4(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Q4 values: Q4 = FY - (Q1 + Q2 + Q3)"""
        df = df.copy()
        q4_rows = []
        
        # Group by metric and year to calculate Q4
        grouped = df.groupby(['Financial Section', 'Section', 'Metric', 'Currency'])
        
        for name, group in grouped:
            # Extract year from Report Period End Date
            group = group.copy()
            group['Year'] = pd.to_datetime(group['Report Period End Date']).dt.year
            
            for year in group['Year'].unique():
                year_data = group[group['Year'] == year]
                
                # Find FY and quarterly data
                fy_data = year_data[year_data['Period'].str.contains('FY', case=False, na=False)]
                q1_data = year_data[year_data['Period'].str.contains('Q1', case=False, na=False)]
                q2_data = year_data[year_data['Period'].str.contains('Q2', case=False, na=False)]
                q3_data = year_data[year_data['Period'].str.contains('Q3', case=False, na=False)]
                
                # Check if we have all required data
                if len(fy_data) > 0 and len(q1_data) > 0 and len(q2_data) > 0 and len(q3_data) > 0:
                    fy_value = fy_data.iloc[0]['Value_Millions']
                    q1_value = q1_data.iloc[0]['Value_Millions']
                    q2_value = q2_data.iloc[0]['Value_Millions']
                    q3_value = q3_data.iloc[0]['Value_Millions']
                    
                    if not pd.isna(fy_value) and not pd.isna(q1_value) and not pd.isna(q2_value) and not pd.isna(q3_value):
                        q4_value = fy_value - (q1_value + q2_value + q3_value)
                        
                        # Create Q4 row based on FY row
                        q4_row = fy_data.iloc[0].copy()
                        q4_row['Period'] = f"Q4 {year}"
                        q4_row['Value_Millions'] = q4_value
                        q4_row['Value'] = q4_value  # Update original value too
                        q4_rows.append(q4_row)
                    else:
                        logger.warning(f"Missing values for Q4 calculation: {name}, Year {year}")
                else:
                    missing = []
                    if len(fy_data) == 0:
                        missing.append('FY')
                    if len(q1_data) == 0:
                        missing.append('Q1')
                    if len(q2_data) == 0:
                        missing.append('Q2')
                    if len(q3_data) == 0:
                        missing.append('Q3')
                    
                    if missing:
                        logger.warning(f"Missing periods for Q4 calculation: {name}, Year {year}, Missing: {', '.join(missing)}")
        
        if q4_rows:
            q4_df = pd.DataFrame(q4_rows)
            df = pd.concat([df, q4_df], ignore_index=True)
            logger.info(f"Added {len(q4_rows)} calculated Q4 rows")
        
        return df
    
    def aggregate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data at section level"""
        # For now, we'll keep all detail but ensure proper grouping
        # Aggregation happens during visualization based on user selection
        return df
    
    def process_data(self, file_paths: List[str]) -> pd.DataFrame:
        """Main processing pipeline"""
        logger.info("Starting data processing pipeline")
        
        # Load data
        df = self.load_csv_files(file_paths)
        if df.empty:
            logger.error("No data loaded")
            return df
        
        # Add company info from filenames
        companies = []
        for fp in file_paths:
            ticker, filing_type, quarters = self.extract_company_info(Path(fp).name)
            companies.append({'file': fp, 'ticker': ticker, 'filing_type': filing_type})
        
        # Processing steps
        df = self.fill_missing_sections(df)
        df = self.normalize_all_values(df)
        df = self.remove_duplicates(df)
        df = self.calculate_q4(df)
        df = self.aggregate_data(df)
        
        self.processed_data = df
        logger.info(f"Processing complete: {len(df)} rows")
        
        return df


class SECVisualizationApp:
    """Streamlit app for SEC data visualization"""
    
    def __init__(self):
        self.processor = SECDataProcessor()
        
    def run(self):
        st.set_page_config(page_title="SEC Filings Analyzer", layout="wide")
        st.title("📊 SEC Filings Data Visualization")
        st.markdown("---")
        
        # File upload
        st.sidebar.header("📁 Data Upload")
        uploaded_files = st.sidebar.file_uploader(
            "Upload CSV files (10-Q, 10-K)",
            type=['csv'],
            accept_multiple_files=True
        )
        
        if not uploaded_files:
            st.info("👈 Please upload CSV files from the sidebar to begin")
            st.markdown("""
            ### Expected CSV Format:
            - **Filename pattern**: `TICKER_10-Q_15-25.csv` or `TICKER_10-K_15-25.csv`
            - **Required columns**: Filing Date, Report Period End Date, Financial Section, 
              Section, Metric, Period, Value, Currency, Units
            """)
            return
        
        # Save uploaded files temporarily and process
        file_paths = []
        for uploaded_file in uploaded_files:
            file_paths.append(uploaded_file)
        
        with st.spinner("Processing data..."):
            # Read files directly from uploaded file objects
            dfs = []
            file_metadata = []  # Store metadata separately
            for file_obj in file_paths:
                df = pd.read_csv(file_obj)
                if 'URL' in df.columns:
                    df = df.drop('URL', axis=1)
                # Add source filename
                df['Source_File'] = file_obj.name
                dfs.append(df)
                
                # Store metadata for later
                ticker, filing_type, quarters = self.processor.extract_company_info(file_obj.name)
                file_metadata.append({
                    'filename': file_obj.name,
                    'ticker': ticker,
                    'filing_type': filing_type,
                    'source_type': 'Quarterly' if filing_type == '10-Q' else 'Annual' if filing_type == '10-K' else 'Unknown'
                })
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                df = self.processor.fill_missing_sections(combined_df)
                df = self.processor.normalize_all_values(df)
                df = self.processor.remove_duplicates(df)
                df = self.processor.calculate_q4(df)
                
                # Add metadata columns after processing
                df['Company'] = df['Source_File'].map({m['filename']: m['ticker'] for m in file_metadata})
                df['Filing_Type'] = df['Source_File'].map({m['filename']: m['filing_type'] for m in file_metadata})
                df['Source_Type'] = df['Source_File'].map({m['filename']: m['source_type'] for m in file_metadata})
                
                self.processor.processed_data = df
                
                st.success(f"✅ Processed {len(df)} rows from {len(uploaded_files)} files")
            else:
                st.error("Failed to load data")
                return
        
        # Extract unique companies from filenames
        companies = []
        for file_obj in uploaded_files:
            ticker, filing_type, quarters = self.processor.extract_company_info(file_obj.name)
            if ticker:
                companies.append(ticker)
        companies = sorted(list(set(companies)))
        
        # Display data summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Companies", len(companies) if companies else "N/A")
        with col3:
            st.metric("Periods", df['Period'].nunique())
        with col4:
            st.metric("Financial Sections", df['Financial Section'].nunique())
        
        st.markdown("---")
        
        # Filters
        st.sidebar.header("🔍 Filters")
        
        # Company selection
        if companies:
            selected_companies = st.sidebar.multiselect(
                "Select Companies",
                options=companies,
                default=companies
            )
            
            # Filter by selected companies
            if selected_companies:
                df = df[df['Source_File'].str.contains('|'.join(selected_companies))]
        
        # Source Type filter (Quarterly vs Annual)
        source_types = sorted(df['Source_Type'].dropna().unique())
        selected_source_types = st.sidebar.multiselect(
            "Select Source Type",
            options=source_types,
            default=source_types
        )
        
        if selected_source_types:
            df = df[df['Source_Type'].isin(selected_source_types)]
        
        if df.empty:
            st.warning("No data available after applying Source Type filter")
            return
        
        # Period filter
        periods = sorted(df['Period'].dropna().unique())
        selected_periods = st.sidebar.multiselect(
            "Select Periods",
            options=periods,
            default=periods
        )
        
        if selected_periods:
            df = df[df['Period'].isin(selected_periods)]
        
        if df.empty:
            st.warning("No data available after applying Period filter")
            return
        
        # Financial Section selection
        financial_sections = sorted(df['Financial Section'].dropna().unique())
        selected_financial_section = st.sidebar.selectbox(
            "Select Financial Section",
            options=financial_sections
        )
        
        # Filter by financial section
        df_filtered = df[df['Financial Section'] == selected_financial_section]
        
        # Section selection
        sections = sorted(df_filtered['Section'].dropna().unique())
        selected_sections = st.sidebar.multiselect(
            "Select Sections",
            options=sections,
            default=sections[:3] if len(sections) >= 3 else sections
        )
        
        if not selected_sections:
            st.warning("Please select at least one section")
            return
        
        # Filter by sections
        df_filtered = df_filtered[df_filtered['Section'].isin(selected_sections)]
        
        # Metric selection
        metrics = sorted(df_filtered['Metric'].dropna().unique())
        selected_metrics = st.sidebar.multiselect(
            "Select Metrics",
            options=metrics,
            default=metrics[:5] if len(metrics) >= 5 else metrics
        )
        
        if not selected_metrics:
            st.warning("Please select at least one metric")
            return
        
        # Filter by metrics
        df_filtered = df_filtered[df_filtered['Metric'].isin(selected_metrics)]
        
        # Visualization
        st.header(f"📈 {selected_financial_section}")
        
        if df_filtered.empty:
            st.warning("No data available for selected filters")
            return
        
        # Create visualizations for each metric
        for metric in selected_metrics:
            metric_data = df_filtered[df_filtered['Metric'] == metric]
            
            if metric_data.empty:
                continue
            
            st.subheader(metric)
            
            # Create bar chart
            fig = go.Figure()
            
            # Group by period, company, and source type
            for source_file in metric_data['Source_File'].unique():
                file_data = metric_data[metric_data['Source_File'] == source_file]
                
                # Extract company name and source type
                ticker = file_data['Company'].iloc[0] if 'Company' in file_data.columns else None
                source_type = file_data['Source_Type'].iloc[0] if 'Source_Type' in file_data.columns else None
                
                label = f"{ticker} ({source_type})" if ticker and source_type else (ticker if ticker else source_file)
                
                # Sort by period
                file_data = file_data.sort_values('Report Period End Date')
                
                fig.add_trace(go.Bar(
                    x=file_data['Period'],
                    y=file_data['Value_Millions'],
                    name=label,
                    text=file_data['Value_Millions'].round(2),
                    textposition='auto',
                ))
            
            fig.update_layout(
                barmode='group',
                xaxis_title="Period",
                yaxis_title="Value (Millions)",
                hovermode='x unified',
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            with st.expander("View Data Table"):
                display_cols = ['Company', 'Source_Type', 'Period', 'Value_Millions', 'Currency', 'Filing Date', 'Report Period End Date']
                available_cols = [col for col in display_cols if col in metric_data.columns]
                st.dataframe(
                    metric_data[available_cols].sort_values('Report Period End Date'),
                    use_container_width=True
                )
        
        # Raw data view
        st.markdown("---")
        with st.expander("🔍 View All Filtered Data"):
            st.dataframe(df_filtered, use_container_width=True)


if __name__ == "__main__":
    app = SECVisualizationApp()
    app.run()