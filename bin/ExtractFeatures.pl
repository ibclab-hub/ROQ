#!/usr/bin/perl
#
use strict;
use warnings;
use Cwd 'abs_path';
use FindBin qw($Bin);
use File::Basename;



my $input = shift or die "Usage: $0 input.bam|input.sam|input.bed\n";
my $bed_file;

if ($input =~ /\.bam$/) {
    my $sam_file = basename($input, ".bam") . ".sam";
    system("samtools view -h $input > $sam_file") == 0 or die "Failed to convert BAM to SAM: $!";
    print STDERR "Converted BAM to SAM: $sam_file\n";

    $bed_file = basename($sam_file, ".sam") . ".bed";
    system("sam2bed < $sam_file > $bed_file") == 0 or die "Failed to convert SAM to BED: $!";
    print STDERR "Converted SAM to BED: $bed_file\n";
} elsif ($input =~ /\.sam$/) {
    $bed_file = basename($input, ".sam") . ".bed";
    system("sam2bed < $input > $bed_file") == 0 or die "Failed to convert SAM to BED: $!";
    print  STDERR "Converted SAM to BED: $bed_file\n";
} elsif ($input =~ /\.bed$/) {
    $bed_file = $input;
    print  STDERR "Using existing BED file: $bed_file\n";
} else {
    die "Invalid input format. Please provide a BAM, SAM, or BED file.\n";
}
my %hs_basecnt = ();

# Tags we ultimately want to populate, in output order.
my @init_tags = ("AS","XS","XM","XO","XG","NM","YS");

# Per-aligner naming quirks we know about so far:
#   - HISAT2 emits "ZS" instead of bowtie2's "XS" (secondary alignment score)
#   - STAR emits "nM" (paired-read mismatch count) which is NOT the same
#     metric as the standard single-read "NM" (edit distance) tag and must
#     never be substituted for it
#   - BWA-MEM and STAR do not emit XM/XO/XG (mismatch/gap-open/gap-ext) or
#     YS (mate alignment score) at all; XM/XO/XG are recovered from the
#     CIGAR string below, YS has no CIGAR-derivable equivalent and is left NA

my %hs_multipleid = ();

open(F, $bed_file) or die "Failed to open BED file: $bed_file\n";
while(<F>){
    chomp;
    if ($_ =~ /^@/){next;}
    my @ar_tmp = split(/\t/,$_);
    my $chr = shift(@ar_tmp);
    my $start = shift(@ar_tmp);
    my $end = shift(@ar_tmp);
    my $id = shift(@ar_tmp);
    my $mapq = shift(@ar_tmp);
    my $order = shift(@ar_tmp);
    my $flag = shift(@ar_tmp);
    my $cigar = shift(@ar_tmp);
    my $mrnm = shift(@ar_tmp);
    my $mpos = shift(@ar_tmp);
    my $isize = shift(@ar_tmp);
    my $read = shift(@ar_tmp);
    my $qual = shift(@ar_tmp);

    # Skip reads HISAT2 (or any splice-aware aligner run without a
    # WGS-only flag) mistakenly treated as spliced: a large N operation
    # in the CIGAR represents an intron-skip, which has no meaning for
    # genomic DNA and can carry spuriously inflated/differently-computed
    # tags (e.g. XS:A: splice-strand instead of a numeric score).
    if ($cigar =~ /\d+N/) {
        print STDERR "SKIPPED (spliced/intron-skip CIGAR, not valid for WGS): $id\t$cigar\n";
        next;
    }

    $hs_multipleid{$id}++;
    my $alnid = $id;

    # raw tag values as emitted by whichever aligner produced this file
    my %hs_tag_raw = ();
    foreach my $thistag (@ar_tmp){
        my ($tag, $fmt, $val) = split(/:/,$thistag);
        # Guard against tag-name collisions with unrelated SAM tags: e.g.
        # HISAT2 emits "XS:A:+/-" (splice strand) on reads with a large
        # N (intron-skip) CIGAR op, which is a completely different tag
        # from the numeric secondary-alignment-score "XS:i:" we want.
        # Only accept numeric (i/f) typed values here.
        next if (defined $fmt && $fmt !~ /^[if]$/);
        $hs_tag_raw{$tag} = $val;
    }

    # populate the canonical feature-tag set, defaulting to NA
    my %hs_tag = ();
    foreach my $thistag (@init_tags){
        $hs_tag{$thistag} = exists($hs_tag_raw{$thistag}) ? $hs_tag_raw{$thistag} : "NA";
    }

    # HISAT2: no XS tag, but ZS carries the same "next-best alignment score" meaning
    if ($hs_tag{"XS"} eq "NA" && exists $hs_tag_raw{"ZS"}) {
        $hs_tag{"XS"} = $hs_tag_raw{"ZS"};
    }

    # BWA / STAR: XM, XO, XG are not emitted at all -> recover gap counts
    # from the CIGAR string, and back out mismatches from NM when NM is
    # itself available (it is not, for STAR's default output; STAR's "nM"
    # is a different, paired-level metric and is intentionally never used
    # here as a substitute for NM/XM)
    if ($hs_tag{"XO"} eq "NA" && $hs_tag{"XG"} eq "NA") {
        my ($cig_gap_opens, $cig_gap_ext, $cig_indel_bases) = parse_cigar_gaps($cigar);
        $hs_tag{"XO"} = $cig_gap_opens;
        $hs_tag{"XG"} = $cig_gap_ext;
        if ($hs_tag{"XM"} eq "NA" && $hs_tag{"NM"} ne "NA" && is_number($hs_tag{"NM"})) {
            $hs_tag{"XM"} = $hs_tag{"NM"} - $cig_indel_bases;
        }
    }

    print "$alnid\t$mapq"; #1
    foreach my $thistag (@init_tags){ #2~8
        print "\t$hs_tag{$thistag}";
    }
    my $diff = "NA";
    if ($hs_tag{"XS"} ne "NA" && $hs_tag{"AS"} ne "NA"
        && is_number($hs_tag{"XS"}) && is_number($hs_tag{"AS"})){
        $diff = abs($hs_tag{"XS"}-$hs_tag{"AS"});
    }
    print "\t$diff";  #9
    print "\t$isize";  #10
    my $gc = GC($read);
    print "\t$gc"; #11
    my ($lowq,$avgq) = QUAL($qual);
    print "\t$lowq";#12
    print "\t$avgq"; #13
    print "\n";
}
close(F);


sub is_number {
    my $v = shift;
    return 0 if (!defined $v);
    return ($v =~ /^-?\d+(\.\d+)?$/) ? 1 : 0;
}

sub GC {
    my $str = shift;
    %hs_basecnt = ("A" => 0 , "T" => 0 , "G" => 0 , "C" => 0 );
    foreach my $this (split(//,$str)){
        $hs_basecnt{$this} ++;
    }
	my $gc = 0;
	if(!($hs_basecnt{"A"}+$hs_basecnt{"C"}+$hs_basecnt{"G"}+$hs_basecnt{"T"}) == 0){
    $gc = ($hs_basecnt{"G"}+$hs_basecnt{"C"})/($hs_basecnt{"A"}+$hs_basecnt{"C"}+$hs_basecnt{"G"}+$hs_basecnt{"T"});

	}
    $gc = sprintf("%.3f", $gc);
    return $gc;
}

sub QUAL {
    my $seq = shift;
    my $PHRED_OFFSET = 33; # Sanger/Illumina 1.8+ quality encoding offset
    my $total = 0;
    my $len = 0;
    my $cnt_under20 = 0;
    foreach my $b (split(//,$seq)){
        my $q = ord($b) - $PHRED_OFFSET;
        $total += $q;
        $len ++;
        if ($q < 20){
            $cnt_under20 ++;
        }
    }
    my $avrg = ($len > 0) ? ($total / $len) : 0;
    $avrg = sprintf("%.3f", $avrg);

    return ($cnt_under20, $avrg)
}

# Derive gap-open count, gap-extension count, and total indel bases directly
# from a CIGAR string. Used as a fallback for aligners (BWA, STAR) that do
# not emit bowtie2-style XM/XO/XG tags.
#   gap_opens        = number of I/D operations (each gap counts once)
#   total_indel_bases = sum of lengths of all I/D operations
#   gap_ext          = total_indel_bases - gap_opens
#                       (bases beyond the first base of each gap, matching
#                       bowtie2's XO+XG = total indel length convention)
sub parse_cigar_gaps {
    my $cigar = shift;
    my $gap_opens = 0;
    my $total_indel_bases = 0;
    return (0, 0, 0) if (!defined $cigar || $cigar eq "*" || $cigar eq "NA");
    while ($cigar =~ /(\d+)([MIDNSHP=X])/g) {
        my ($len, $op) = ($1, $2);
        if ($op eq 'I' || $op eq 'D') {
            $gap_opens++;
            $total_indel_bases += $len;
        }
    }
    my $gap_ext = $total_indel_bases - $gap_opens;
    return ($gap_opens, $gap_ext, $total_indel_bases);
}
