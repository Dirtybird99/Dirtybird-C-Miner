#!/bin/bash
#
# A/B Benchmark Harness for SA Text Prefetch Validation
#
# Runs statistically rigorous benchmarks to detect ~1% performance difference
# from the SA text prefetch optimization with high confidence.
#
# Expected gain: ~0.7-1.2%
# At 17.6 KH/s, 1% = ~176 H/s
#
# Runs alternating tests to minimize thermal/time-of-day bias.
#
# Usage: ./ab_prefetch_bench.sh [options]
#
# Options:
#   -i, --iterations N      Number of iterations per variant (default: 10)
#   -d, --duration N        Mining duration per run in seconds (default: 30)
#   -t, --threads N         Number of mining threads (default: auto-detect)
#   -a, --daemon-address    DERO daemon address (default: 203.0.113.10)
#   -p, --port N            DERO daemon port (default: 10100)
#   -w, --wallet ADDR       DERO wallet address
#   -c, --cooldown N        Cooldown between runs in seconds (default: 5)
#   -s, --skip-build        Skip building and use existing binaries
#   --disable-partition     Also disable partition prefetch
#   -h, --help              Show this help message

set -e

# Default configuration
ITERATIONS=10
DURATION=30
THREADS=0  # 0 = auto-detect
DAEMON_ADDRESS="203.0.113.10"
PORT=10100
WALLET="DERO_WALLET_PLACEHOLDER"
COOLDOWN_SECONDS=5
SKIP_BUILD=false
DISABLE_PARTITION_PREFETCH=false

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Build directories
BUILD_WITH_PREFETCH="$PROJECT_DIR/build"
BUILD_WITHOUT_PREFETCH="$PROJECT_DIR/build-noprefetch"

# Results arrays (bash arrays)
declare -a RESULTS_WITH_PREFETCH
declare -a RESULTS_WITHOUT_PREFETCH

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -i, --iterations N      Number of iterations per variant (default: 10)"
    echo "  -d, --duration N        Mining duration per run in seconds (default: 30)"
    echo "  -t, --threads N         Number of mining threads (default: auto-detect)"
    echo "  -a, --daemon-address    DERO daemon address (default: 203.0.113.10)"
    echo "  -p, --port N            DERO daemon port (default: 10100)"
    echo "  -w, --wallet ADDR       DERO wallet address"
    echo "  -c, --cooldown N        Cooldown between runs in seconds (default: 5)"
    echo "  -s, --skip-build        Skip building and use existing binaries"
    echo "  --disable-partition     Also disable partition prefetch"
    echo "  -h, --help              Show this help message"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        -a|--daemon-address)
            DAEMON_ADDRESS="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -w|--wallet)
            WALLET="$2"
            shift 2
            ;;
        -c|--cooldown)
            COOLDOWN_SECONDS="$2"
            shift 2
            ;;
        -s|--skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --disable-partition)
            DISABLE_PARTITION_PREFETCH=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Auto-detect threads if not specified
if [[ $THREADS -eq 0 ]]; then
    if [[ -f /proc/cpuinfo ]]; then
        THREADS=$(grep -c ^processor /proc/cpuinfo)
    else
        THREADS=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    fi
    if [[ $THREADS -gt 20 ]]; then
        THREADS=20
    fi
    echo "Auto-detected $THREADS threads"
fi

write_banner() {
    local text="$1"
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN} $text${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""
}

build_variant() {
    local build_dir="$1"
    local name="$2"
    local extra_flags="$3"

    write_banner "Building $name variant"

    # Clean and create build directory
    if [[ -d "$build_dir" ]]; then
        echo "Cleaning existing build directory..."
        rm -rf "$build_dir"
    fi
    mkdir -p "$build_dir"

    # Prepare CMake flags
    local cmake_flags="-G Ninja -DCMAKE_BUILD_TYPE=Release"

    if [[ -n "$extra_flags" ]]; then
        cmake_flags="$cmake_flags -DCMAKE_C_FLAGS_RELEASE=\"$extra_flags\" -DCMAKE_CXX_FLAGS_RELEASE=\"$extra_flags\""
        echo "Extra flags: $extra_flags"
    fi

    # Run CMake
    echo "Running CMake..."
    pushd "$build_dir" > /dev/null

    eval cmake -S "$PROJECT_DIR" -B . $cmake_flags
    if [[ $? -ne 0 ]]; then
        echo "CMake configuration failed"
        exit 1
    fi

    echo "Building..."
    ninja
    if [[ $? -ne 0 ]]; then
        echo "Build failed"
        exit 1
    fi

    popd > /dev/null
    echo -e "${GREEN}Build complete: $name${NC}"
}

get_miner_path() {
    local build_dir="$1"

    local paths=(
        "$build_dir/bin/dirtybird-miner-cpu"
        "$build_dir/dirtybird-miner-cpu"
    )

    for path in "${paths[@]}"; do
        if [[ -x "$path" ]]; then
            echo "$path"
            return 0
        fi
    done

    echo "Miner executable not found in $build_dir" >&2
    exit 1
}

run_benchmark() {
    local miner_path="$1"
    local name="$2"

    echo -e "${YELLOW}Running $name benchmark ($DURATION seconds)...${NC}"

    local output
    output=$("$miner_path" \
        --dero \
        --daemon-address "$DAEMON_ADDRESS" \
        --port "$PORT" \
        --wallet "$WALLET" \
        --threads "$THREADS" \
        --mine-time "$DURATION" 2>&1)

    # Parse hashrate from output
    local hashrate=0

    # Try to find final/average hashrate
    if echo "$output" | grep -qiE "(final|average|total).*[0-9]+\.?[0-9]*\s*KH/s"; then
        hashrate=$(echo "$output" | grep -oiE "(final|average|total).*?([0-9]+\.?[0-9]*)\s*KH/s" | grep -oE "[0-9]+\.?[0-9]*" | tail -1)
        hashrate=$(echo "$hashrate * 1000" | bc -l)
    elif echo "$output" | grep -qiE "(final|average|total).*[0-9]+\s*H/s"; then
        hashrate=$(echo "$output" | grep -oiE "(final|average|total).*?([0-9]+)\s*H/s" | grep -oE "[0-9]+" | tail -1)
    # Fallback: find any hashrate
    elif echo "$output" | grep -qE "[0-9]+\.?[0-9]*\s*KH/s"; then
        hashrate=$(echo "$output" | grep -oE "[0-9]+\.?[0-9]*\s*KH/s" | tail -1 | grep -oE "[0-9]+\.?[0-9]*")
        hashrate=$(echo "$hashrate * 1000" | bc -l)
    elif echo "$output" | grep -qE "[0-9]+\s*H/s"; then
        hashrate=$(echo "$output" | grep -oE "[0-9]+\s*H/s" | tail -1 | grep -oE "[0-9]+")
    fi

    if [[ "$hashrate" == "0" || -z "$hashrate" ]]; then
        echo -e "${RED}WARNING: Could not parse hashrate from output${NC}"
        echo "Output snippet: ${output:0:500}"
    fi

    echo "$hashrate"
}

calculate_statistics() {
    local -n values=$1
    local n=${#values[@]}

    if [[ $n -eq 0 ]]; then
        echo "0 0 0 0 0"
        return
    fi

    # Calculate mean
    local sum=0
    for v in "${values[@]}"; do
        sum=$(echo "$sum + $v" | bc -l)
    done
    local mean=$(echo "$sum / $n" | bc -l)

    # Calculate variance and stddev
    local variance=0
    for v in "${values[@]}"; do
        local diff=$(echo "$v - $mean" | bc -l)
        variance=$(echo "$variance + ($diff * $diff)" | bc -l)
    done
    variance=$(echo "$variance / ($n - 1)" | bc -l)
    local stddev=$(echo "sqrt($variance)" | bc -l)

    # Calculate min and max
    local min=${values[0]}
    local max=${values[0]}
    for v in "${values[@]}"; do
        if (( $(echo "$v < $min" | bc -l) )); then
            min=$v
        fi
        if (( $(echo "$v > $max" | bc -l) )); then
            max=$v
        fi
    done

    echo "$mean $stddev $variance $min $max"
}

calculate_ttest() {
    local mean1=$1
    local var1=$2
    local n1=$3
    local mean2=$4
    local var2=$5
    local n2=$6

    # Welch's t-test
    local se=$(echo "sqrt($var1/$n1 + $var2/$n2)" | bc -l)
    if [[ $(echo "$se == 0" | bc -l) -eq 1 ]]; then
        echo "0 1 0"
        return
    fi

    local t=$(echo "($mean1 - $mean2) / $se" | bc -l)

    # Welch-Satterthwaite degrees of freedom
    local num=$(echo "($var1/$n1 + $var2/$n2)^2" | bc -l)
    local denom=$(echo "(($var1/$n1)^2)/($n1-1) + (($var2/$n2)^2)/($n2-1)" | bc -l)
    local df=$(echo "$num / $denom" | bc -l)

    # Approximate p-value
    local abs_t=$(echo "if ($t < 0) -$t else $t" | bc -l)
    local p_value

    if (( $(echo "$df > 30" | bc -l) )); then
        # Normal approximation for large df
        local z=$(echo "$abs_t / sqrt(2)" | bc -l)
        p_value=$(echo "e(-$z * $z)" | bc -l)
    else
        # Conservative lookup
        if (( $(echo "$abs_t > 4.0" | bc -l) )); then
            p_value="0.001"
        elif (( $(echo "$abs_t > 3.5" | bc -l) )); then
            p_value="0.002"
        elif (( $(echo "$abs_t > 3.0" | bc -l) )); then
            p_value="0.005"
        elif (( $(echo "$abs_t > 2.75" | bc -l) )); then
            p_value="0.01"
        elif (( $(echo "$abs_t > 2.5" | bc -l) )); then
            p_value="0.02"
        elif (( $(echo "$abs_t > 2.2" | bc -l) )); then
            p_value="0.05"
        elif (( $(echo "$abs_t > 2.0" | bc -l) )); then
            p_value="0.1"
        elif (( $(echo "$abs_t > 1.5" | bc -l) )); then
            p_value="0.2"
        else
            p_value="0.5"
        fi
    fi

    echo "$t $p_value $df"
}

format_hashrate() {
    local hashrate=$1

    if (( $(echo "$hashrate >= 1000" | bc -l) )); then
        printf "%.2f KH/s" $(echo "$hashrate / 1000" | bc -l)
    else
        printf "%.0f H/s" $hashrate
    fi
}

# ============================================================================
# Main Script
# ============================================================================

write_banner "A/B Prefetch Benchmark Harness"

echo "Configuration:"
echo "  Iterations:   $ITERATIONS per variant"
echo "  Duration:     $DURATION seconds per run"
echo "  Threads:      $THREADS"
echo "  Daemon:       ${DAEMON_ADDRESS}:${PORT}"
echo "  Cooldown:     $COOLDOWN_SECONDS seconds"
echo ""

# Build both variants
if [[ "$SKIP_BUILD" == "false" ]]; then
    # Build WITH prefetch (default)
    build_variant "$BUILD_WITH_PREFETCH" "WITH prefetch" ""

    # Build WITHOUT prefetch
    NO_PREFETCH_FLAGS="-DDISABLE_SA_TEXT_PREFETCH"
    if [[ "$DISABLE_PARTITION_PREFETCH" == "true" ]]; then
        NO_PREFETCH_FLAGS="$NO_PREFETCH_FLAGS -DDISABLE_PARTITION_PREFETCH"
    fi
    build_variant "$BUILD_WITHOUT_PREFETCH" "WITHOUT prefetch" "$NO_PREFETCH_FLAGS"
fi

# Get miner paths
MINER_WITH_PREFETCH=$(get_miner_path "$BUILD_WITH_PREFETCH")
MINER_WITHOUT_PREFETCH=$(get_miner_path "$BUILD_WITHOUT_PREFETCH")

echo "Miner WITH prefetch:    $MINER_WITH_PREFETCH"
echo "Miner WITHOUT prefetch: $MINER_WITHOUT_PREFETCH"

write_banner "Starting A/B Benchmark"
echo "Total runs: $((ITERATIONS * 2)) (alternating)"
echo ""

# Run alternating benchmarks
for ((i = 0; i < ITERATIONS; i++)); do
    run_num=$((i + 1))
    echo ""
    echo -e "${MAGENTA}=== Iteration $run_num of $ITERATIONS ===${NC}"

    # Run WITH prefetch
    hashrate=$(run_benchmark "$MINER_WITH_PREFETCH" "WITH prefetch")
    RESULTS_WITH_PREFETCH+=("$hashrate")
    echo -e "  ${GREEN}WITH prefetch: $(format_hashrate $hashrate)${NC}"

    # Cooldown
    echo "  Cooling down for $COOLDOWN_SECONDS seconds..."
    sleep $COOLDOWN_SECONDS

    # Run WITHOUT prefetch
    hashrate=$(run_benchmark "$MINER_WITHOUT_PREFETCH" "WITHOUT prefetch")
    RESULTS_WITHOUT_PREFETCH+=("$hashrate")
    echo -e "  ${YELLOW}WITHOUT prefetch: $(format_hashrate $hashrate)${NC}"

    # Cooldown (except on last iteration)
    if [[ $i -lt $((ITERATIONS - 1)) ]]; then
        echo "  Cooling down for $COOLDOWN_SECONDS seconds..."
        sleep $COOLDOWN_SECONDS
    fi
done

# Calculate statistics
write_banner "A/B Prefetch Benchmark Results"

read mean1 stddev1 var1 min1 max1 <<< $(calculate_statistics RESULTS_WITH_PREFETCH)
read mean2 stddev2 var2 min2 max2 <<< $(calculate_statistics RESULTS_WITHOUT_PREFETCH)
read t_stat p_value df <<< $(calculate_ttest $mean1 $var1 $ITERATIONS $mean2 $var2 $ITERATIONS)

echo -e "${GREEN}WITH Text Prefetch:${NC}"
echo "  Mean:     $(format_hashrate $mean1)"
echo "  StdDev:   $(format_hashrate $stddev1)"
echo "  Min:      $(format_hashrate $min1)"
echo "  Max:      $(format_hashrate $max1)"
echo "  Runs:     $ITERATIONS"
echo ""

echo -e "${YELLOW}WITHOUT Text Prefetch:${NC}"
echo "  Mean:     $(format_hashrate $mean2)"
echo "  StdDev:   $(format_hashrate $stddev2)"
echo "  Min:      $(format_hashrate $min2)"
echo "  Max:      $(format_hashrate $max2)"
echo "  Runs:     $ITERATIONS"
echo ""

difference=$(echo "$mean1 - $mean2" | bc -l)
if (( $(echo "$mean2 > 0" | bc -l) )); then
    percent_diff=$(echo "($difference / $mean2) * 100" | bc -l)
else
    percent_diff=0
fi

echo -e "${CYAN}Difference:${NC}"
if (( $(echo "$difference >= 0" | bc -l) )); then
    sign="+"
else
    sign=""
fi
printf "  %s%s (%s%.2f%%)\n" "$sign" "$(format_hashrate $difference)" "$sign" "$percent_diff"
echo ""

echo -e "${CYAN}Statistical Analysis:${NC}"
printf "  T-statistic:        %.3f\n" $t_stat
printf "  p-value:            %.4f\n" $p_value
printf "  Degrees of freedom: %.1f\n" $df
echo ""

# Determine significance
if (( $(echo "$p_value < 0.01" | bc -l) )); then
    echo -e "${GREEN}Result: SIGNIFICANT at p<0.01${NC}"
elif (( $(echo "$p_value < 0.05" | bc -l) )); then
    echo -e "${GREEN}Result: SIGNIFICANT at p<0.05${NC}"
elif (( $(echo "$p_value < 0.10" | bc -l) )); then
    echo -e "${YELLOW}Result: MARGINALLY SIGNIFICANT at p<0.10${NC}"
else
    echo -e "${RED}Result: NOT SIGNIFICANT (p >= 0.10)${NC}"
fi

echo ""
if (( $(echo "$difference > 0" | bc -l) )); then
    printf "${GREEN}Conclusion: Text prefetch provides ~%.1f%% improvement${NC}\n" "$percent_diff"
elif (( $(echo "$difference < 0" | bc -l) )); then
    abs_percent=$(echo "if ($percent_diff < 0) -$percent_diff else $percent_diff" | bc -l)
    printf "${RED}Conclusion: Text prefetch causes ~%.1f%% regression${NC}\n" "$abs_percent"
else
    echo -e "${YELLOW}Conclusion: No measurable difference${NC}"
fi

# Output raw data
echo ""
echo -e "${GRAY}Raw Data:${NC}"
echo -e "${GRAY}  WITH prefetch:    ${RESULTS_WITH_PREFETCH[*]}${NC}"
echo -e "${GRAY}  WITHOUT prefetch: ${RESULTS_WITHOUT_PREFETCH[*]}${NC}"

# Save results to JSON file
RESULTS_FILE="$PROJECT_DIR/ab_benchmark_results.json"
cat > "$RESULTS_FILE" << EOF
{
  "timestamp": "$(date '+%Y-%m-%d %H:%M:%S')",
  "configuration": {
    "iterations": $ITERATIONS,
    "duration": $DURATION,
    "threads": $THREADS,
    "daemon_address": "$DAEMON_ADDRESS",
    "port": $PORT
  },
  "with_prefetch": {
    "hashrates": [${RESULTS_WITH_PREFETCH[*]}],
    "mean": $mean1,
    "stddev": $stddev1
  },
  "without_prefetch": {
    "hashrates": [${RESULTS_WITHOUT_PREFETCH[*]}],
    "mean": $mean2,
    "stddev": $stddev2
  },
  "analysis": {
    "difference": $difference,
    "percent_difference": $percent_diff,
    "t_statistic": $t_stat,
    "p_value": $p_value,
    "degrees_of_freedom": $df
  }
}
EOF

echo ""
echo -e "${CYAN}Results saved to: $RESULTS_FILE${NC}"
