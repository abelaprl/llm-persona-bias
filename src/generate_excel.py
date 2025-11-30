#!/usr/bin/env python3
"""
Convert CSV results to Excel with formatting.
"""
import csv
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from openpyxl.utils import get_column_letter

OUTPUT_DIR = Path("results/reports")

def csv_to_excel(csv_path: Path, xlsx_path: Path):
    """Convert CSV to Excel with formatting."""
    wb = Workbook()
    ws = wb.active
    ws.title = csv_path.stem[:31]  # Excel sheet name max 31 chars
    
    # Read CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if not rows:
        return
    
    # Write data
    for row_idx, row in enumerate(rows, 1):
        for col_idx, value in enumerate(row, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            
            # Try to convert to number
            try:
                cell.value = float(value)
            except:
                pass
    
    # Style header
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")
    thin_border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    for col_idx in range(1, len(rows[0]) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center')
        cell.border = thin_border
    
    # Auto-adjust column widths
    for col_idx in range(1, len(rows[0]) + 1):
        max_length = 0
        column_letter = get_column_letter(col_idx)
        for row in rows:
            if col_idx <= len(row):
                max_length = max(max_length, len(str(row[col_idx - 1])))
        ws.column_dimensions[column_letter].width = min(max_length + 2, 50)
    
    # Add borders to all cells
    for row_idx in range(2, len(rows) + 1):
        for col_idx in range(1, len(rows[0]) + 1):
            ws.cell(row=row_idx, column=col_idx).border = thin_border
    
    # Freeze header row
    ws.freeze_panes = 'A2'
    
    wb.save(xlsx_path)
    print(f"  Saved: {xlsx_path}")


def create_summary_excel():
    """Create formatted summary Excel with multiple sheets."""
    wb = Workbook()
    
    # Summary sheet
    ws_summary = wb.active
    ws_summary.title = "Summary"
    
    # Read summary CSV
    summary_csv = OUTPUT_DIR / "summary_all_models.csv"
    if summary_csv.exists():
        with open(summary_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        for row_idx, row in enumerate(rows, 1):
            for col_idx, value in enumerate(row, 1):
                cell = ws_summary.cell(row=row_idx, column=col_idx)
                try:
                    cell.value = float(value)
                except:
                    cell.value = value
        
        # Style header
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        for col_idx in range(1, len(rows[0]) + 1):
            cell = ws_summary.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', wrap_text=True)
            cell.border = thin_border
        
        # Highlight accuracy column
        accuracy_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
        for row_idx in range(2, len(rows) + 1):
            ws_summary.cell(row=row_idx, column=6).fill = accuracy_fill  # accuracy_pct column
            for col_idx in range(1, len(rows[0]) + 1):
                ws_summary.cell(row=row_idx, column=col_idx).border = thin_border
        
        # Auto-adjust column widths
        for col_idx in range(1, len(rows[0]) + 1):
            max_length = 0
            column_letter = get_column_letter(col_idx)
            for row in rows:
                if col_idx <= len(row):
                    max_length = max(max_length, len(str(row[col_idx - 1])))
            ws_summary.column_dimensions[column_letter].width = min(max_length + 2, 25)
        
        ws_summary.freeze_panes = 'A2'
    
    # Save
    xlsx_path = OUTPUT_DIR / "summary_all_models.xlsx"
    wb.save(xlsx_path)
    print(f"  Saved: {xlsx_path}")


def main():
    print("="*60)
    print("Converting CSV to Excel")
    print("="*60)
    
    # Convert individual model CSVs
    csv_files = [
        "bert_nebulon_results.csv",
        "grok_4_1_results.csv",
        "nemotron-nano_results.csv",
    ]
    
    for csv_file in csv_files:
        csv_path = OUTPUT_DIR / csv_file
        if csv_path.exists():
            xlsx_path = csv_path.with_suffix('.xlsx')
            print(f"\nConverting {csv_file}...")
            csv_to_excel(csv_path, xlsx_path)
    
    # Create summary Excel
    print("\nCreating summary Excel...")
    create_summary_excel()
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
