import os
import re

def parse_file(file_path, mime_type, original_name):
    """
    Parses a file and returns a list of raw segment dicts:
    [{
        'content': str,
        'page_number': int | None,
        'sheet_name': str | None,
        'start_line': int | None,
        'end_line': int | None
    }]
    """
    ext = os.path.splitext(original_name)[1].lower()
    segments = []

    if ext == '.pdf':
        segments = _parse_pdf(file_path)
    elif ext == '.docx':
        segments = _parse_docx(file_path)
    elif ext in ['.csv', '.xlsx', '.xls']:
        segments = _parse_excel_csv(file_path, ext)
    else:
        # Default text/code parser
        segments = _parse_text_file(file_path)

    return segments


def _parse_pdf(file_path):
    segments = []
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(file_path)
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ''
            text = text.strip()
            if text:
                segments.append({
                    'content': text,
                    'page_number': page_idx + 1,
                    'sheet_name': None,
                    'start_line': None,
                    'end_line': None
                })
    except Exception as e:
        print(f"[FileParser] PDF error: {e}")
    return segments


def _parse_docx(file_path):
    segments = []
    try:
        import docx
        doc = docx.Document(file_path)
        current_block = []
        current_line_count = 0
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                current_block.append(text)
                current_line_count += 1
                if current_line_count >= 15:
                    segments.append({
                        'content': "\n".join(current_block),
                        'page_number': None,
                        'sheet_name': None,
                        'start_line': None,
                        'end_line': None
                    })
                    current_block = []
                    current_line_count = 0
        if current_block:
            segments.append({
                'content': "\n".join(current_block),
                'page_number': None,
                'sheet_name': None,
                'start_line': None,
                'end_line': None
            })
    except Exception as e:
        print(f"[FileParser] DOCX error: {e}")
    return segments


def _parse_excel_csv(file_path, ext):
    segments = []
    try:
        import pandas as pd
        if ext == '.csv':
            df = pd.read_csv(file_path)
            sheets = {'CSV_Data': df}
        else:
            sheets = pd.read_excel(file_path, sheet_name=None)

        for sheet_name, df in sheets.items():
            if df.empty:
                continue
            # Convert dataframe rows to CSV strings in chunks of 50 rows
            total_rows = len(df)
            chunk_size = 50
            for i in range(0, total_rows, chunk_size):
                sub_df = df.iloc[i:i+chunk_size]
                csv_str = sub_df.to_csv(index=False)
                segments.append({
                    'content': f"Sheet: {sheet_name} (Rows {i+1} to {min(i+chunk_size, total_rows)})\n" + csv_str,
                    'page_number': None,
                    'sheet_name': sheet_name,
                    'start_line': i + 1,
                    'end_line': min(i + chunk_size, total_rows)
                })
    except Exception as e:
        print(f"[FileParser] Excel/CSV error: {e}")
    return segments


def _parse_text_file(file_path):
    segments = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        chunk_lines = 50
        total_lines = len(lines)
        for i in range(0, total_lines, chunk_lines):
            block = lines[i:i+chunk_lines]
            text = "".join(block).strip()
            if text:
                segments.append({
                    'content': text,
                    'page_number': None,
                    'sheet_name': None,
                    'start_line': i + 1,
                    'end_line': min(i + chunk_lines, total_lines)
                })
    except Exception as e:
        print(f"[FileParser] Text file error: {e}")
    return segments
