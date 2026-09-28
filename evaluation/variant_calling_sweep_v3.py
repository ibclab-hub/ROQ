"""
Variant calling threshold sweep with bcftools (v2).

Compares MAPQ-based vs ROQ-based read filtering for variant calling accuracy
against the GIAB HG001 truth set.

Fixes applied (v2):
  1. COMBINED filter: reads without RQ tag are now skipped (not passed through)
  2. GIAB high-confidence BED: TP/FP/FN counted only within confident regions
  3. pysam.fetch() coordinates: 0-based half-open (r_s -= 1)
  4. Truth normalization: done once at start, reused for all conditions
  5. Baseline (NONE): unfiltered baseline added
  6. n_reads + retention_rate: tracked for each condition
  7. -Ou: uncompressed BCF streaming for mpileup

Input:
  bam:         input BAM (with RQ tags for ROQ filtering)
  truth_vcf:   GIAB truth set VCF (bgzipped+indexed)
  fa:          reference FASTA
  region:      e.g. "chr1:1-10000000"
  bed:         GIAB high-confidence region BED (recommended)

Output:
  CSV with columns:
    filter_type, threshold, n_reads, retention_rate,
    n_variants, tp, fp, fn, precision, recall, f1
"""
import sys
import os
import argparse
import subprocess
import tempfile
import pandas as pd

BCFTOOLS = os.environ.get('BCFTOOLS', 'bcftools')
SAMTOOLS = os.environ.get('SAMTOOLS', 'samtools')


def run(cmd, check=True):
    print(f"  $ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"  STDERR: {r.stderr[:500]}")
        raise RuntimeError(f"Command failed: {cmd}")
    return r


def count_reads(bam_path, region=None):
    """Count reads in BAM, optionally restricted to region."""
    cmd = f"{SAMTOOLS} view -c {bam_path}"
    if region:
        cmd += f" {region}"
    return int(run(cmd).stdout.strip())


def parse_region(region):
    """Parse 'chr1:1-10000000' into (chrom, start_0based, end)."""
    r_chr, r_range = region.split(':')
    r_s, r_e = r_range.split('-')
    # pysam.fetch() is 0-based half-open: [start, end)
    # region string is 1-based: chr1:1-10000000 means [1, 10000000]
    # So fetch(chr, 0, 10000000) covers the same interval
    return r_chr, int(r_s) - 1, int(r_e)


def filter_bam_none(in_bam, out_bam, region):
    """No filtering — keep all reads in region (baseline)."""
    run(f"{SAMTOOLS} view -b {in_bam} {region} > {out_bam}")
    run(f"{SAMTOOLS} index {out_bam}")


def filter_bam_mapq(in_bam, out_bam, threshold, region):
    """Filter by MAPQ using samtools view -q."""
    run(f"{SAMTOOLS} view -b -q {threshold} {in_bam} {region} > {out_bam}")
    run(f"{SAMTOOLS} index {out_bam}")


def filter_bam_roq(in_bam, out_bam, threshold, region):
    """Filter by ROQ tag using pysam. Reads without ZQ tag are skipped.

    NOTE: tag is "ZQ", not "RQ" -- PacBio HiFi/CCS BAMs carry a native
    "RQ:f:" tag (predicted CCS read quality) that predROQ.py's ROQ tag
    would otherwise collide with, so predROQ.py now writes ZQ instead.
    """
    import pysam
    r_chr, r_s, r_e = parse_region(region)
    sf = pysam.AlignmentFile(in_bam, 'rb')
    of = pysam.AlignmentFile(out_bam, 'wb', template=sf)
    n = 0
    for read in sf.fetch(r_chr, r_s, r_e):
        if not read.has_tag('ZQ'):
            continue
        if read.get_tag('ZQ') >= threshold:
            of.write(read)
            n += 1
    sf.close()
    of.close()
    run(f"{SAMTOOLS} index {out_bam}")
    return n


def filter_bam_combined(in_bam, out_bam, mapq_t, roq_t, region):
    """Combined MAPQ AND ROQ filter. Reads without ZQ tag are skipped."""
    import pysam
    r_chr, r_s, r_e = parse_region(region)
    sf = pysam.AlignmentFile(in_bam, 'rb')
    of = pysam.AlignmentFile(out_bam, 'wb', template=sf)
    n = 0
    for read in sf.fetch(r_chr, r_s, r_e):
        if read.mapping_quality < mapq_t:
            continue
        # reads without ZQ tag are skipped (not passed through)
        if not read.has_tag('ZQ'):
            continue
        if read.get_tag('ZQ') < roq_t:
            continue
        of.write(read)
        n += 1
    sf.close()
    of.close()
    run(f"{SAMTOOLS} index {out_bam}")
    return n


def call_variants(in_bam, out_vcf, fa, region):
    """bcftools mpileup + call. Uses -Ou for uncompressed BCF streaming."""
    run(f"{BCFTOOLS} mpileup -Ou -f {fa} -r {region} {in_bam} | "
        f"{BCFTOOLS} call -mv -Ov -o {out_vcf}")


def sort_bed(bed_in, bed_out):
    """Sort BED file by chrom then start position (bcftools view -R requires this)."""
    run(f"sort -k1,1 -k2,2n {bed_in} > {bed_out}")


def normalize_vcf(vcf_in, vcf_out, fa):
    """Normalize VCF (split multiallelics, left-align indels).
    Output is bgzipped + tabixed."""
    run(f"{BCFTOOLS} norm -f {fa} -m -both {vcf_in} | bgzip > {vcf_out} && tabix -p vcf {vcf_out}")


def mask_to_bed(vcf_in, vcf_out, bed):
    """Restrict bgzipped+tabixed VCF to BED regions.
    Input must be indexed. Output is bgzipped + tabixed."""
    run(f"{BCFTOOLS} view -R {bed} {vcf_in} | bgzip > {vcf_out} && tabix -p vcf {vcf_out}")


def normalize_and_mask(vcf_in, vcf_out, fa, bed=None):
    """Normalize VCF and optionally restrict to BED regions.
    Two-step process: norm → bgzip → tabix → view -R → bgzip → tabix.
    (bcftools view -R cannot read from a pipe — it needs an indexed file.)"""
    norm_out = vcf_out + '.norm_tmp.gz'
    normalize_vcf(vcf_in, norm_out, fa)
    if bed:
        mask_to_bed(norm_out, vcf_out, bed)
        os.remove(norm_out)
        # Clean up tabix index of temp
        if os.path.exists(norm_out + '.tbi'):
            os.remove(norm_out + '.tbi')
    else:
        # Just rename the normalized file
        os.rename(norm_out, vcf_out)
        if os.path.exists(norm_out + '.tbi'):
            os.rename(norm_out + '.tbi', vcf_out + '.tbi')


def eval_calls(calls_vcf, truth_norm_bed, outdir, fa, bed=None):
    """Compare calls with truth set using bcftools isec.
    Returns (tp, fp, fn, n_variants).

    truth_norm_bed: pre-normalized (and BED-masked) truth VCF path.
    """
    # Normalize calls (and mask to BED if provided)
    calls_norm = os.path.join(outdir, 'calls.norm.gz')
    normalize_and_mask(calls_vcf, calls_norm, fa, bed)

    # isec: 0000 = calls-only (FP), 0001 = truth-only (FN), 0002 = shared (TP)
    isec_dir = os.path.join(outdir, 'isec')
    run(f"{BCFTOOLS} isec -p {isec_dir} {calls_norm} {truth_norm_bed}")

    fp = int(run(f"{BCFTOOLS} view {isec_dir}/0000.vcf | grep -v '^#' | wc -l").stdout.strip() or 0)
    fn = int(run(f"{BCFTOOLS} view {isec_dir}/0001.vcf | grep -v '^#' | wc -l").stdout.strip() or 0)
    tp = int(run(f"{BCFTOOLS} view {isec_dir}/0002.vcf | grep -v '^#' | wc -l").stdout.strip() or 0)
    n_var = int(run(f"{BCFTOOLS} view {calls_norm} | grep -v '^#' | wc -l").stdout.strip() or 0)
    return tp, fp, fn, n_var


def main():
    ap = argparse.ArgumentParser(
        description='Variant calling threshold sweep: MAPQ vs ROQ vs COMBINED')
    ap.add_argument('--bam', required=True, help='BAM with RQ tags')
    ap.add_argument('--truth', required=True, help='GIAB truth VCF (bgzipped+indexed)')
    ap.add_argument('--fa', required=True, help='Reference FASTA')
    ap.add_argument('--region', required=True, help='e.g. chr1:1-10000000')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--bed', default=None,
                    help='GIAB high-confidence region BED (recommended)')
    ap.add_argument('--mapq-thresholds', default='0,10,20,30,40,50,60',
                    help='Comma-separated MAPQ thresholds')
    # ROQ is capped at 42 (maxMAPQ in calculate_roq), so 50/60 always retain
    # 0% of reads -- not useful defaults. ROQ predictions also tend to
    # saturate near the top of the range, so finer-grained thresholds near
    # 40-42 resolve more than the same coarse grid used for MAPQ. Floats
    # are allowed (e.g. 41.5).
    ap.add_argument('--roq-thresholds', default='10,20,30,35,38,40,41,41.5,41.8',
                    help='Comma-separated ROQ thresholds (floats allowed)')
    ap.add_argument('--combined', action='store_true',
                    help='Include COMBINED (MAPQ AND ROQ) filter')
    ap.add_argument('--combined-thresholds', default='10,20,30,40',
                    help='Comma-separated thresholds for COMBINED (uses same value for MAPQ and ROQ)')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Sanity check: make sure --bam was actually tagged by the (fixed)
    # predROQ.py before sinking time into the full sweep. Catches the case
    # of accidentally pointing --bam at an untagged BAM, or one tagged by
    # an older predROQ.py version that still used the "RQ" tag name.
    print("\n=== Checking for ZQ tag in --bam ===")
    import pysam as _pysam
    _check = _pysam.AlignmentFile(args.bam, 'rb')
    _has_zq = False
    _has_old_rq = False
    for _i, _read in enumerate(_check):
        if _read.has_tag('ZQ'):
            _has_zq = True
            break
        if _read.has_tag('RQ'):
            _has_old_rq = True
        if _i > 10000:
            break
    _check.close()
    if not _has_zq:
        msg = f"No ZQ tag found in the first reads of {args.bam}."
        if _has_old_rq:
            msg += (" Found an 'RQ' tag instead -- this BAM looks like it was "
                    "tagged by an OLDER version of predROQ.py (pre tag-rename), "
                    "or it's a PacBio BAM with its own native RQ:f: quality tag. "
                    "Re-run the current predROQ.py to (re)tag with ZQ before "
                    "using this script.")
        else:
            msg += " Run predROQ.py on this BAM first."
        sys.exit(f"ERROR: {msg}")
    print("  OK: ZQ tag found.")
    bed_sorted = None
    if args.bed:
        bed_sorted = os.path.join(args.outdir, 'confident_regions.sorted.bed')
        print(f"\n=== Sorting BED file ===")
        sort_bed(args.bed, bed_sorted)
        print(f"  Sorted BED: {bed_sorted}")

    # ------------------------------------------------------------------
    # Step 1: Pre-normalize truth VCF once (and mask to BED if provided)
    # ------------------------------------------------------------------
    print("\n=== Pre-normalizing truth VCF ===")
    truth_norm = os.path.join(args.outdir, 'truth.norm.gz')
    normalize_and_mask(args.truth, truth_norm, args.fa, bed_sorted)
    n_truth = int(run(f"{BCFTOOLS} view {truth_norm} | grep -v '^#' | wc -l").stdout.strip())
    print(f"  Truth variants after norm + BED mask: {n_truth}")

    # ------------------------------------------------------------------
    # Step 2: Count total reads in region (for retention rate baseline)
    # ------------------------------------------------------------------
    print("\n=== Counting total reads in region ===")
    n_total = count_reads(args.bam, args.region)
    print(f"  Total reads in {args.region}: {n_total}")

    # ------------------------------------------------------------------
    # Step 3: Run threshold sweep
    # ------------------------------------------------------------------
    results = []

    # --- Baseline (NONE) ---
    print(f"\n=== NONE (no filtering) ===")
    tmpdir = tempfile.mkdtemp(prefix='vc_NONE_', dir=args.outdir)
    filt_bam = os.path.join(tmpdir, 'filtered.bam')
    filter_bam_none(args.bam, filt_bam, args.region)
    n_reads = count_reads(filt_bam)
    calls_vcf = os.path.join(tmpdir, 'calls.vcf')
    call_variants(filt_bam, calls_vcf, args.fa, args.region)
    tp, fp, fn, n_var = eval_calls(calls_vcf, truth_norm, tmpdir, args.fa, bed_sorted)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    results.append({
        'filter_type': 'NONE', 'threshold': 0,
        'n_reads': n_reads, 'retention_rate': round(n_reads / n_total, 4) if n_total else 0,
        'n_variants': n_var, 'tp': tp, 'fp': fp, 'fn': fn,
        'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4),
    })
    print(f"  n_reads={n_reads} retention={n_reads/n_total:.4f} TP={tp} FP={fp} FN={fn} "
          f"prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

    # --- MAPQ sweep ---
    mapq_ts = [int(t) for t in args.mapq_thresholds.split(',')]
    for t in mapq_ts:
        if t == 0:
            continue  # threshold=0 is same as NONE baseline
        print(f"\n=== MAPQ >= {t} ===")
        tmpdir = tempfile.mkdtemp(prefix=f'vc_MAPQ_{t}_', dir=args.outdir)
        filt_bam = os.path.join(tmpdir, 'filtered.bam')
        filter_bam_mapq(args.bam, filt_bam, t, args.region)
        n_reads = count_reads(filt_bam)
        calls_vcf = os.path.join(tmpdir, 'calls.vcf')
        call_variants(filt_bam, calls_vcf, args.fa, args.region)
        tp, fp, fn, n_var = eval_calls(calls_vcf, truth_norm, tmpdir, args.fa, bed_sorted)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        results.append({
            'filter_type': 'MAPQ', 'threshold': t,
            'n_reads': n_reads, 'retention_rate': round(n_reads / n_total, 4) if n_total else 0,
            'n_variants': n_var, 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4),
        })
        print(f"  n_reads={n_reads} retention={n_reads/n_total:.4f} TP={tp} FP={fp} FN={fn} "
              f"prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

    # --- ROQ sweep ---
    roq_ts = [float(t) for t in args.roq_thresholds.split(',')]
    for t in roq_ts:
        if t == 0:
            continue  # threshold=0 is same as NONE baseline (for reads with RQ tag)
        print(f"\n=== ROQ >= {t} ===")
        tmpdir = tempfile.mkdtemp(prefix=f'vc_ROQ_{t}_', dir=args.outdir)
        filt_bam = os.path.join(tmpdir, 'filtered.bam')
        filter_bam_roq(args.bam, filt_bam, t, args.region)
        n_reads = count_reads(filt_bam)
        calls_vcf = os.path.join(tmpdir, 'calls.vcf')
        call_variants(filt_bam, calls_vcf, args.fa, args.region)
        tp, fp, fn, n_var = eval_calls(calls_vcf, truth_norm, tmpdir, args.fa, bed_sorted)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        results.append({
            'filter_type': 'ROQ', 'threshold': t,
            'n_reads': n_reads, 'retention_rate': round(n_reads / n_total, 4) if n_total else 0,
            'n_variants': n_var, 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4),
        })
        print(f"  n_reads={n_reads} retention={n_reads/n_total:.4f} TP={tp} FP={fp} FN={fn} "
              f"prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

    # --- COMBINED sweep (optional) ---
    if args.combined:
        comb_ts = [int(t) for t in args.combined_thresholds.split(',')]
        for t in comb_ts:
            print(f"\n=== COMBINED (MAPQ>={t} AND ROQ>={t}) ===")
            tmpdir = tempfile.mkdtemp(prefix=f'vc_COMBINED_{t}_', dir=args.outdir)
            filt_bam = os.path.join(tmpdir, 'filtered.bam')
            filter_bam_combined(args.bam, filt_bam, t, t, args.region)
            n_reads = count_reads(filt_bam)
            calls_vcf = os.path.join(tmpdir, 'calls.vcf')
            call_variants(filt_bam, calls_vcf, args.fa, args.region)
            tp, fp, fn, n_var = eval_calls(calls_vcf, truth_norm, tmpdir, args.fa, bed_sorted)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            results.append({
                'filter_type': 'COMBINED', 'threshold': t,
                'n_reads': n_reads, 'retention_rate': round(n_reads / n_total, 4) if n_total else 0,
                'n_variants': n_var, 'tp': tp, 'fp': fp, 'fn': fn,
                'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4),
            })
            print(f"  n_reads={n_reads} retention={n_reads/n_total:.4f} TP={tp} FP={fp} FN={fn} "
                  f"prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

    # ------------------------------------------------------------------
    # Step 4: Save results
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)
    outcsv = os.path.join(args.outdir, 'variant_calling_sweep.csv')
    df.to_csv(outcsv, index=False)
    print(f"\nSaved to {outcsv}")
    print(df.to_string())

    # ------------------------------------------------------------------
    # Step 5: Generate summary (MAPQ vs ROQ comparison)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY: MAPQ vs ROQ")
    print("=" * 80)

    summary_rows = []

    # Baseline
    none_row = df[df.filter_type == 'NONE']
    if not none_row.empty:
        n = none_row.iloc[0]
        print(f"\nBaseline (NONE):  reads={n.n_reads}  variants={n.n_variants}  "
              f"TP={n.tp} FP={n.fp} FN={n.fn}  prec={n.precision} rec={n.recall} F1={n.f1}")
        summary_rows.append({
            'comparison': 'NONE (baseline)', 'threshold': 0,
            'mapq_fp': n.fp, 'roq_fp': n.fp, 'fp_reduction': 0, 'fp_reduction_pct': 0,
            'mapq_f1': n.f1, 'roq_f1': n.f1, 'f1_improvement': 0,
            'mapq_retention': n.retention_rate, 'roq_retention': n.retention_rate,
        })

    # Per-threshold comparison
    mapq_ts = [int(t) for t in args.mapq_thresholds.split(',') if t != '0']
    roq_ts = [float(t) for t in args.roq_thresholds.split(',') if t != '0']
    common_ts = sorted(set(mapq_ts) & set(roq_ts))

    print(f"\n{'T':>4} {'MAPQ_FP':>8} {'ROQ_FP':>8} {'FP_red':>8} {'FP_%':>7} "
          f"{'MAPQ_F1':>8} {'ROQ_F1':>8} {'dF1':>7} "
          f"{'MAPQ_ret':>9} {'ROQ_ret':>9}")
    print("-" * 80)

    for t in common_ts:
        m = df[(df.filter_type == 'MAPQ') & (df.threshold == t)]
        r = df[(df.filter_type == 'ROQ') & (df.threshold == t)]
        if m.empty or r.empty:
            continue
        m = m.iloc[0]
        r = r.iloc[0]
        fp_red = m.fp - r.fp
        fp_pct = 100 * fp_red / m.fp if m.fp > 0 else 0
        d_f1 = r.f1 - m.f1
        print(f"{t:>4} {m.fp:>8} {r.fp:>8} {fp_red:>8} {fp_pct:>6.1f}% "
              f"{m.f1:>8.4f} {r.f1:>8.4f} {d_f1:>+7.4f} "
              f"{m.retention_rate:>9.4f} {r.retention_rate:>9.4f}")
        summary_rows.append({
            'comparison': f'MAPQ>={t} vs ROQ>={t}', 'threshold': t,
            'mapq_fp': m.fp, 'roq_fp': r.fp,
            'fp_reduction': fp_red, 'fp_reduction_pct': round(fp_pct, 1),
            'mapq_f1': m.f1, 'roq_f1': r.f1, 'f1_improvement': round(d_f1, 4),
            'mapq_retention': m.retention_rate, 'roq_retention': r.retention_rate,
        })

    # COMBINED summary if present
    comb_df = df[df.filter_type == 'COMBINED']
    if not comb_df.empty:
        print(f"\n--- COMBINED (MAPQ AND ROQ) ---")
        print(f"{'T':>4} {'FP':>8} {'F1':>8} {'retention':>10}")
        print("-" * 40)
        for _, c in comb_df.iterrows():
            print(f"{c.threshold:>4} {c.fp:>8} {c.f1:>8.4f} {c.retention_rate:>10.4f}")

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.outdir, 'summary_mapq_vs_roq.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary saved to {summary_csv}")


if __name__ == '__main__':
    main()
