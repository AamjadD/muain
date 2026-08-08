import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / 'data' / 'raw' / 'cases.csv'
OUTPUT_CLEAN = BASE_DIR / 'data' / 'processed' / 'cleaned_cases.csv'
OUTPUT_TRAIN = BASE_DIR / 'data' / 'processed' / 'cases_train.csv'
OUTPUT_TEST = BASE_DIR / 'data' / 'processed' / 'cases_test.csv'


def normalize_section(value: str) -> str:
    if pd.isna(value):
        return value
    value = str(value).strip()
    replacements = {
        'القضايا الجنائية': 'قضايا جنائية',
    }
    return replacements.get(value, value)

#Merging the case and the ruling

def build_case_text(row: pd.Series) -> str:
    parts = []
    for col in ['الدعوى', 'الحكم']:
        val = row.get(col)
        if pd.notna(val):
            text = str(val).strip()
            if text:
                parts.append(text)
    return '\n\n'.join(parts)


def main() -> None:
    OUTPUT_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    # Drop spreadsheet-export artifacts like Unnamed columns.
    unnamed_cols = [c for c in df.columns if str(c).startswith('Unnamed:')]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Remove completely empty rows.
    df = df.dropna(how='all').copy()

    # Standardize section labels.
    if 'القسم' in df.columns:
        df['القسم'] = df['القسم'].apply(normalize_section)

    # Keep only rows that are usable for supervised text classification.
    required_cols = ['القسم', 'الفرع', 'السند الشرعي', 'الدعوى']
    for col in required_cols:
        df[col] = df[col].replace(r'^\s*$', pd.NA, regex=True)
    df = df.dropna(subset=required_cols).copy()

    # Create a stable unique identifier even if original case number duplicates exist.
    df = df.reset_index(drop=True)
    df.insert(0, 'case_uid', [f'case_{i+1:04d}' for i in range(len(df))])

    # Create the text field used later by the model.
    df['case_text'] = df.apply(build_case_text, axis=1)
    df['case_text'] = df['case_text'].replace(r'^\s*$', pd.NA, regex=True)
    df = df.dropna(subset=['case_text']).copy()

    # Reorder columns for convenience.
    preferred_order = ['case_uid', 'الرقم', 'القسم', 'الفرع', 'السند الشرعي', 'case_text', 'الدعوى', 'الحكم', 'الاستئناف']
    existing_order = [c for c in preferred_order if c in df.columns]
    remaining = [c for c in df.columns if c not in existing_order]
    df = df[existing_order + remaining]

    # Save cleaned dataset.
    df.to_csv(OUTPUT_CLEAN, index=False, encoding='utf-8-sig')

    # Stratified split by main section/category.
    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df['القسم']
    )

    # Sort for stable output files.
    train_df = train_df.sort_values('case_uid').reset_index(drop=True)
    test_df = test_df.sort_values('case_uid').reset_index(drop=True)

    train_df.to_csv(OUTPUT_TRAIN, index=False, encoding='utf-8-sig')
    test_df.to_csv(OUTPUT_TEST, index=False, encoding='utf-8-sig')

    print('Done.')
    print(f'Cleaned dataset: {OUTPUT_CLEAN} -> {df.shape}')
    print(f'Train dataset:   {OUTPUT_TRAIN} -> {train_df.shape}')
    print(f'Test dataset:    {OUTPUT_TEST} -> {test_df.shape}')
    print('\nClass distribution (cleaned):')
    print(df['القسم'].value_counts())
    print('\nClass distribution (train):')
    print(train_df['القسم'].value_counts())
    print('\nClass distribution (test):')
    print(test_df['القسم'].value_counts())


if __name__ == '__main__':
    main()
