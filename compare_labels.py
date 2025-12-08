#!/usr/bin/env python3
"""
Compare two label CSV files to identify which labels changed.

This helps document label corrections and understand their impact.
"""

import pandas as pd
from pathlib import Path


def compare_label_files(old_csv: str, new_csv: str):
    """
    Compare two label files and show which labels changed.

    Parameters
    ----------
    old_csv : str
        Path to old labels CSV
    new_csv : str
        Path to new (corrected) labels CSV
    """

    print("="*80)
    print("LABEL COMPARISON REPORT")
    print("="*80)

    # Load both files
    old_df = pd.read_csv(old_csv)
    new_df = pd.read_csv(new_csv)

    print(f"\nOld labels: {old_csv}")
    print(f"New labels: {new_csv}")

    # Find merge keys
    potential_keys = ['trial', 'fly', 'trial_id', 'sample_id']
    merge_keys = [k for k in potential_keys if k in old_df.columns and k in new_df.columns]

    if not merge_keys:
        print("\n❌ ERROR: No common identifier columns found!")
        print(f"Old columns: {list(old_df.columns)}")
        print(f"New columns: {list(new_df.columns)}")
        return

    print(f"\nMerging on: {merge_keys}")

    # Find label column
    label_cols = ['label', 'response', 'reaction', 'class']
    old_label_col = None
    new_label_col = None

    for col in label_cols:
        if col in old_df.columns:
            old_label_col = col
        if col in new_df.columns:
            new_label_col = col

    if not old_label_col or not new_label_col:
        print("\n❌ ERROR: No label column found!")
        print(f"Old columns: {list(old_df.columns)}")
        print(f"New columns: {list(new_df.columns)}")
        return

    print(f"Label column: '{old_label_col}' (old) → '{new_label_col}' (new)")

    # Merge
    merged = old_df.merge(
        new_df,
        on=merge_keys,
        how='outer',
        suffixes=('_old', '_new'),
        indicator=True
    )

    # Handle column naming after merge
    if old_label_col == new_label_col:
        old_col = f"{old_label_col}_old"
        new_col = f"{new_label_col}_new"
    else:
        old_col = old_label_col
        new_col = new_label_col

    # Find changes
    changed = merged[merged[old_col] != merged[new_col]].copy()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\nTotal trials in old: {len(old_df)}")
    print(f"Total trials in new: {len(new_df)}")
    print(f"Total matched: {len(merged[merged['_merge'] == 'both'])}")

    # Check for missing trials
    only_old = len(merged[merged['_merge'] == 'left_only'])
    only_new = len(merged[merged['_merge'] == 'right_only'])

    if only_old > 0:
        print(f"\n⚠️  {only_old} trials only in OLD file (removed?)")
    if only_new > 0:
        print(f"\n⚠️  {only_new} trials only in NEW file (added?)")

    print(f"\n{'='*80}")
    print(f"LABEL CHANGES: {len(changed)} trials changed")
    print(f"{'='*80}")

    if len(changed) == 0:
        print("\n✅ No label changes detected!")
        return

    # Count change types
    changes_0_to_1 = len(changed[(changed[old_col] == 0) & (changed[new_col] == 1)])
    changes_1_to_0 = len(changed[(changed[old_col] == 1) & (changed[new_col] == 0)])

    print(f"\nChanges 0→1 (added reactions): {changes_0_to_1}")
    print(f"Changes 1→0 (removed reactions): {changes_1_to_0}")

    # Reaction rate change
    old_reaction_rate = old_df[old_label_col].mean()
    new_reaction_rate = new_df[new_label_col].mean()
    rate_change = new_reaction_rate - old_reaction_rate

    print(f"\nReaction rate change:")
    print(f"  Old: {old_reaction_rate:.2%} ({old_df[old_label_col].sum()}/{len(old_df)} reactions)")
    print(f"  New: {new_reaction_rate:.2%} ({new_df[new_label_col].sum()}/{len(new_df)} reactions)")
    print(f"  Change: {rate_change:+.2%}")

    # Show detailed changes
    print("\n" + "="*80)
    print("DETAILED LABEL CHANGES")
    print("="*80)

    display_cols = merge_keys + [old_col, new_col]

    # Add fly column if not already in merge keys
    if 'fly' in changed.columns and 'fly' not in merge_keys:
        display_cols.insert(1, 'fly')

    # Add label_intensity if available
    if 'label_intensity_old' in changed.columns:
        display_cols.append('label_intensity_old')
    if 'label_intensity_new' in changed.columns:
        display_cols.append('label_intensity_new')

    available_cols = [c for c in display_cols if c in changed.columns]

    # Sort by change type
    changed_sorted = changed.sort_values([old_col, new_col])

    print("\nChanges 0→1 (Added reactions):")
    changes_added = changed_sorted[(changed_sorted[old_col] == 0) & (changed_sorted[new_col] == 1)]
    if len(changes_added) > 0:
        print(changes_added[available_cols].to_string(index=False))
    else:
        print("  None")

    print("\nChanges 1→0 (Removed reactions):")
    changes_removed = changed_sorted[(changed_sorted[old_col] == 1) & (changed_sorted[new_col] == 0)]
    if len(changes_removed) > 0:
        print(changes_removed[available_cols].to_string(index=False))
    else:
        print("  None")

    # Save to file
    output_path = Path("label_changes_detailed.csv")
    changed[available_cols].to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"Detailed changes saved to: {output_path}")
    print(f"{'='*80}")

    # By fly analysis
    if 'fly' in changed.columns:
        print("\n" + "="*80)
        print("CHANGES BY FLY")
        print("="*80)

        fly_changes = changed.groupby('fly').size().sort_values(ascending=False)
        print("\nFlies with most label changes:")
        for fly, count in fly_changes.head(10).items():
            fly_total = len(old_df[old_df['fly'] == fly]) if 'fly' in old_df.columns else "?"
            print(f"  {fly}: {count} changes (out of {fly_total} trials)")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if len(changed) > 50:
        print("\n⚠️  Many labels changed (>50)!")
        print("   → This is a significant revision")
        print("   → Make sure you have clear criteria for all changes")
        print("   → Consider having a second reviewer verify")
    elif len(changed) > 20:
        print("\n⚠️  Moderate number of changes (20-50)")
        print("   → Document reasoning for each change")
        print("   → Review patterns to ensure consistency")
    else:
        print("\n✅ Small number of changes (<20)")
        print("   → Should have minimal impact on model")
        print("   → Document in label_correction_log.md")

    if abs(rate_change) > 0.10:
        print(f"\n⚠️  Large reaction rate change ({rate_change:+.1%})!")
        print("   → This will significantly affect class balance")
        print("   → Verify this is intended")

    print("\nNext steps:")
    print("1. Review the detailed changes above")
    print("2. Document in label_correction_log.md")
    print("3. Retrain model with new labels (use --seed 42)")
    print("4. Compare model performance before/after")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python compare_labels.py <old_labels.csv> <new_labels.csv>")
        print("\nExample:")
        print("  python compare_labels.py \\")
        print("    /path/to/old/scoring_results.csv \\")
        print("    /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv")
        sys.exit(1)

    old_csv = sys.argv[1]
    new_csv = sys.argv[2]

    if not Path(old_csv).exists():
        print(f"❌ ERROR: Old labels file not found: {old_csv}")
        sys.exit(1)

    if not Path(new_csv).exists():
        print(f"❌ ERROR: New labels file not found: {new_csv}")
        sys.exit(1)

    compare_label_files(old_csv, new_csv)
