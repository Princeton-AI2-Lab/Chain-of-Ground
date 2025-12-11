import csv
import os
import re

# Code -> CSV column name
COLUMNS = {
    'AS': 'AS',
    'PYC': 'pycharm',
    'VSC': 'vscode',
    'VM': 'vmware',
    'UE': 'unreal',
    'PS': 'PS',
    'BL': 'BL',
    'PR': 'premiere',
    'DR': 'davinci_macos',
    'AI': 'AI',
    'FL': 'fruitloops',
    'CAD': 'CAD',
    'SW': 'solidworks',
    'INV': 'inventor',
    'QRS': 'quartus',
    'VVD': 'vivado',
    'MAT': 'matlab',
    'ORG': 'origin',
    'EVW': 'eviews',
    'STT': 'stata',
    'PPT': 'powerpoint_windows',
    'EXC': 'excel_macos',
    'WRD': 'word',
    'LNX': 'linux_common',
    'MAC': 'macos_common',
    'WIN': 'windows_common',
}

CATEGORIES = [
    ('development', ['AS', 'PYC', 'VSC', 'VM', 'UE']),
    ('creative', ['PS', 'BL', 'PR', 'DR', 'AI', 'FL']),
    ('cad', ['CAD', 'SW', 'INV', 'QRS', 'VVD']),
    ('scientific', ['MAT', 'ORG', 'EVW', 'STT']),
    ('office', ['PPT', 'EXC', 'WRD']),
    ('operating_systems', ['LNX', 'MAC', 'WIN']),
]

NUM_PATTERN = re.compile(r"[-+]?\d*\.?\d+")

def parse_number(val):
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    m = NUM_PATTERN.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None

def compute_avg(row, cols):
    vals = []
    for code in cols:
        col = COLUMNS[code]
        n = parse_number(row.get(col))
        if n is not None:
            vals.append(n)
    if not vals:
        return ''
    return f"{(sum(vals)/len(vals)):.1f}"

def compute_total_avg(row, fields):
    # Average across all present task columns from all categories
    all_codes = []
    for _, codes in CATEGORIES:
        all_codes.extend(codes)
    vals = []
    for code in all_codes:
        col = COLUMNS[code]
        if col in fields:
            n = parse_number(row.get(col))
            if n is not None:
                vals.append(n)
    if not vals:
        return ''
    return f"{(sum(vals)/len(vals)):.1f}"

def process_file(path, out_dir):
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []

    has_model = 'model' in fields

    # Build output header: only model, per-category averages, and total average
    out_fields = ['model'] if has_model else []
    for cat_name, _ in CATEGORIES:
        out_fields.append(f'avg_{cat_name}')
    out_fields.append('avg_total')

    out_rows = []
    for row in rows:
        out_row = {}
        if has_model:
            out_row['model'] = row.get('model', '')
        for cat_name, codes in CATEGORIES:
            # Average over only present columns for this row
            present_codes = [c for c in codes if COLUMNS[c] in fields]
            out_row[f'avg_{cat_name}'] = compute_avg(row, present_codes)

        # Total average over all present task columns
        out_row['avg_total'] = compute_total_avg(row, fields)

        out_rows.append(out_row)

    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{base}.categories_full.csv")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(out_rows)

    return f"Wrote {os.path.basename(out_path)}: {len(out_rows)} rows"

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Write canonical category outputs here
    out_dir = os.path.join(base_dir, 'classified')

    # Read multi-metric CSV sources from the ordered/ directory
    source_dir = os.path.join(base_dir, 'ordered')
    if not os.path.isdir(source_dir):
        print(f"Source directory not found: {source_dir}")
        return

    csv_files = [fn for fn in os.listdir(source_dir) if fn.endswith('.csv')]
    for fn in sorted(csv_files):
        path = os.path.join(source_dir, fn)
        try:
            print(process_file(path, out_dir))
        except Exception as e:
            print(f"Error processing {fn}: {e}")

    # Also update sspro benchmark if present in root
    sspro = os.path.join(base_dir, 'sspro_benchmark_first_models.csv')
    if os.path.isfile(sspro):
        try:
            print(process_file(sspro, out_dir))
        except Exception as e:
            print(f"Error processing sspro_benchmark_first_models.csv: {e}")

if __name__ == '__main__':
    main()