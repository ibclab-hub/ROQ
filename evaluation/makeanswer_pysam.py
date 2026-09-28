import argparse
import sys

import pysam

parser = argparse.ArgumentParser(
    description='Ground-truth ROR label generation, pysam-based. '
                'Replaces makeanswer.pl for the ANSWER side only: builds the '
                'true-origin lookup directly from the simulator SAM/BAM '
                '(bypassing sam2bed, which was found to silently drop some '
                'reads -- including some with no indels -- causing their '
                'true origin to go missing and their label to default to 0 '
                'even when the read was actually well-aligned). The aligned '
                'side is still walked as a BED file (same file '
                'ExtractFeatures.pl processed), so output stays row-aligned '
                'with the feature table for paste.'
)
parser.add_argument('-answer', '--answer', help='ART-generated ground-truth SAM or BAM '
                                                  '(e.g. paired_dat.sam) -- read directly, '
                                                  'not via sam2bed', required=True)
parser.add_argument('-bed', '--bed', help='aligned-side BED file (e.g. out.bed, the same '
                                           'file ExtractFeatures.pl processed)', required=True)
parser.add_argument('-o', '--output', help='output label file (one ratio per line, '
                                            'row-aligned with --bed for paste)', required=True)
args = parser.parse_args()

# ---- Step 1: build the true-origin lookup directly from the answer SAM/BAM ----
answer_mode = 'rb' if args.answer.endswith('.bam') else 'r'
answer_file = pysam.AlignmentFile(args.answer, answer_mode)

hs_answer = {}
n_answer_total = 0
n_answer_skipped_unmapped = 0

for read in answer_file:
    n_answer_total += 1
    if read.is_unmapped:
        n_answer_skipped_unmapped += 1
        continue

    mate = 1 if read.is_read1 else 2
    composite_id = f"{read.query_name}:{mate}"

    chrom = read.reference_name
    start = read.reference_start          # 0-based, matches sam2bed/BED convention
    end = read.reference_end              # exclusive end, CIGAR-aware

    hs_answer.setdefault(composite_id, []).append((chrom, start, end))

answer_file.close()
print(f"Answer file: {n_answer_total} records, {n_answer_skipped_unmapped} unmapped skipped, "
      f"{len(hs_answer)} unique composite IDs", file=sys.stderr)


def overlap_ratio(cur_chr, cur_st, cur_ed, comp_chr, comp_st, comp_ed):
    if comp_chr != cur_chr:
        return 0.0
    if cur_ed < comp_st:
        return 0.0
    if cur_st > comp_ed:
        return 0.0
    positions = sorted([cur_st, cur_ed, comp_st, comp_ed])
    denom = cur_ed - cur_st
    if denom == 0:
        return 0.0
    return (positions[2] - positions[1]) / denom


import re

n_bed_total = 0
n_skipped_spliced = 0
n_matched = 0
n_warned = 0

with open(args.bed) as bed_f, open(args.output, 'w') as out_f:
    for line in bed_f:
        n_bed_total += 1
        fields = line.rstrip('\n').split('\t')
        cur_chr, cur_st, cur_ed, read_id, mapq, strand, flag, cigar = fields[:8]
        cur_st = int(cur_st)
        cur_ed = int(cur_ed)
        flag = int(flag)

        # must exactly mirror ExtractFeatures.pl's skip condition, so output
        # stays row-for-row aligned with the feature table
        if re.search(r'\d+N', cigar):
            n_skipped_spliced += 1
            print(f"SKIPPED (spliced/intron-skip CIGAR, not valid for WGS): "
                  f"{read_id}\t{cigar}", file=sys.stderr)
            continue

        mate = 1 if (flag & 0x40) else 2
        composite_id = f"{read_id}:{mate}"
        answer_list = hs_answer.get(composite_id)

        max_ratio = 0.0
        if answer_list is not None:
            n_matched += 1
            for comp_chr, comp_st, comp_ed in answer_list:
                r = overlap_ratio(cur_chr, cur_st, cur_ed, comp_chr, comp_st, comp_ed)
                if r > max_ratio:
                    max_ratio = r
        else:
            n_warned += 1
            print(f"WARNING: no answer found for composite_id={composite_id}", file=sys.stderr)

        out_f.write(f"{max_ratio:.3f}\n")

print(f"\nBED rows: {n_bed_total}", file=sys.stderr)
print(f"Skipped (spliced/intron-skip CIGAR): {n_skipped_spliced}", file=sys.stderr)
print(f"Matched to an answer: {n_matched}", file=sys.stderr)
print(f"No answer found (WARNING, label=0): {n_warned}", file=sys.stderr)
print(f"Saved labels to {args.output}", file=sys.stderr)
