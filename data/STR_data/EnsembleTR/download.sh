#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-ensembletr_raw_data}"
TMPDIR="${OUTDIR}/tmp"
mkdir -p "$TMPDIR"

INFO_URL="https://ensemble-tr.s3.us-east-2.amazonaws.com/tables/repeat_tables.zip"
STATS_URL="https://ensemble-tr.s3.us-east-2.amazonaws.com/tables/afreq_tables.zip"

CHROMS=( {1..22} )

download_and_extract() {
    local url="$1" zipname="$2" subdir="$3"
    local zippath="$TMPDIR/$zipname"
    local extractdir="$TMPDIR/$subdir"

    if [[ ! -f "$zippath" ]]; then
        echo "Downloading $url"
        curl -C - -o "$zippath" "$url"
    else
        echo "Already downloaded: $zippath"
    fi

    mkdir -p "$extractdir"
    unzip -o -q "$zippath" -d "$extractdir"
}

merge_in_order() {
    local indir="$1" prefix="$2" output="$3"
    local first=1 files_merged=0

    : > "$output"  # truncate/create

    for chrom in "${CHROMS[@]}"; do
        local f="${indir}/${prefix}_chr${chrom}.csv"
        if [[ ! -f "$f" ]]; then
            echo "  Warning: missing $f, skipping" >&2
            continue
        fi
        if (( first == 1 )); then
            cat "$f" >> "$output"
            first=0
        else
            tail -n +2 "$f" >> "$output"
        fi
        files_merged=$((files_merged + 1))
    done

    echo "Merged $files_merged files into $output ($(wc -l < "$output") lines)"
}

download_and_extract "$INFO_URL"  "repeat_tables.zip" "info"
download_and_extract "$STATS_URL" "afreq_tables.zip"  "stats"

merge_in_order "$TMPDIR/info"  "repeat_info" "$OUTDIR/repeat_info.tsv"
merge_in_order "$TMPDIR/stats" "afreq_het"   "$OUTDIR/afreq_het.tsv"

rm -rf "$TMPDIR"
echo "Done. Outputs in $OUTDIR"