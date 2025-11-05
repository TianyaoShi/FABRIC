#!/bin/bash

# Script to profile all selected tests using the CPU Monitor

# Configuration parameters
NUM_CORES_HOST=64
NUM_CORES_VM=32
TDP=280
# LOOP_TIME=30  # Duration in minutes for each test to run

# List of tests to profile
TESTS=(
#     "mycompress-multi"
#     "mycompress-single"
#     "mydecompress"
#     "myfftw"
# #    "myhpcc"
#     "myjpeg"
# #    "mynpb"
#     "myspark"
#     "myssl"
#     "myvideoenc"
    "parsec.blackscholes" 
    "parsec.bodytrack" 
    "parsec.canneal" 
    "parsec.dedup" 
    "parsec.facesim" 
    "parsec.ferret" 
    "parsec.fluidanimate" 
    "parsec.freqmine" 
    # "parsec.netdedup"  # network programs, out of scope and not compatible with multiple cores
    # "parsec.netferret" 
    # "parsec.netstreamcluster" 
    # "parsec.raytrace" # problematic program, cannot work with n>1
    "parsec.streamcluster" 
    "parsec.swaptions" 
    "parsec.vips" 
    "parsec.x264" 
    "splash2x.barnes" 
    "splash2x.cholesky" 
    "splash2x.fft" 
    "splash2x.fmm" 
    "splash2x.lu_cb" 
    "splash2x.lu_ncb" 
    "splash2x.ocean_cp" 
    "splash2x.ocean_ncp" 
    "splash2x.radiosity" 
    "splash2x.radix" # incompatible with n=48, must be power of 2
    "splash2x.raytrace" 
    "splash2x.volrend" 
    "splash2x.water_nsquared" 
    "splash2x.water_spatial"
)

# Create a results directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
RESULTS_DIR="profile_results"
mkdir -p "$RESULTS_DIR"

# Log file for overall execution
LOG_FILE="${LOG_DIR}/profile_all_${TIMESTAMP}.log"

echo "===================================" | tee -a "$LOG_FILE"
echo "Starting profiling of all tests at: $(date)" | tee -a "$LOG_FILE"
echo "Results will be saved in: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo " CPU and VM specifications:" | tee -a "$LOG_FILE"
echo "  - Host CPU cores: $NUM_CORES_HOST" | tee -a "$LOG_FILE"
echo "  - VM CPU cores: $NUM_CORES_VM" | tee -a "$LOG_FILE"
echo "  - TDP: ${TDP}W" | tee -a "$LOG_FILE"
echo "===================================" | tee -a "$LOG_FILE"

# Run each test
for TEST in "${TESTS[@]}"; do
    for PARALLELISM in 1 2 4 8 16 32 48 64; do # Adjust based on available cores, for c3d-60 use up to 60
        # skip invalid parallelism for certain tests
        if [[ "$TEST" == "splash2x.radix" && ! ( "$PARALLELISM" =~ ^(1|2|4|8|16|32|64)$ ) ]]; then
            continue
        fi
        echo "" | tee -a "$LOG_FILE"
        echo "--------------------------------------" | tee -a "$LOG_FILE"
        # echo "Starting test: $TEST at $(date)" | tee -a "$LOG_FILE"
        echo "Starting test with parallelism: $PARALLELISM at $(date)" | tee -a "$LOG_FILE"
        
        # TEST_COMMAND="TOTAL_LOOP_TIME=$LOOP_TIME phoronix-test-suite batch-run $TEST"
        TEST_COMMAND="parsecmgmt -a run -p $TEST -i native -n $PARALLELISM"
        TEST_LOG="${RESULTS_DIR}/${TEST}_${PARALLELISM}_cores.log"
        
        echo "Command: $TEST_COMMAND" | tee -a "$LOG_FILE"
        echo "Log file: $TEST_LOG" | tee -a "$LOG_FILE"
        
        # Run the test with CPU monitoring
        # python cpu_monitor.py \
        #     --command "$TEST_COMMAND" \
        #     2>&1 | tee "$TEST_LOG"
        python3 cpu_monitor_gcp.py \
            --num_cores_host "$NUM_CORES_HOST" \
            --num_cores_vm "$NUM_CORES_VM" \
            --tdp "$TDP" \
            --command "$TEST_COMMAND" \
            2>&1 | tee "$TEST_LOG"
        
        # Capture exit code
        EXIT_CODE=${PIPESTATUS[0]}
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "Test $TEST completed successfully" | tee -a "$LOG_FILE"
        else
            echo "Warning: Test $TEST exited with code $EXIT_CODE" | tee -a "$LOG_FILE"
        fi
        
        echo "Test $TEST finished at $(date)" | tee -a "$LOG_FILE"
        echo "--------------------------------------" | tee -a "$LOG_FILE"
        
        # Move generated JSON and CSV files to results directory
        find . -type f -name "gcp_cpu_profile_*.json" -newer "$TEST_LOG" -exec mv {} "$RESULTS_DIR"/ \;
        find . -type f -name "idle_power_*.csv" -newer "$TEST_LOG" -exec mv {} "$RESULTS_DIR"/ \;
        find . -type f -name "program_power_*.csv" -newer "$TEST_LOG" -exec mv {} "$RESULTS_DIR"/ \;
        
        # Short pause between tests
        sleep 5
    done
done

echo "" | tee -a "$LOG_FILE"
echo "===================================" | tee -a "$LOG_FILE"
echo "All tests completed at: $(date)" | tee -a "$LOG_FILE"
echo "Results saved in: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "===================================" | tee -a "$LOG_FILE"

# Create a summary of all results
SUMMARY_FILE="${RESULTS_DIR}/summary.txt"
echo "Test Summary ($(date))" > "$SUMMARY_FILE"
echo "============================" >> "$SUMMARY_FILE"

for TEST in "${TESTS[@]}"; do
    for PARALLELISM in 1 2 4 8 16 32 48 64; do
        TEST_NAME="${TEST}_${PARALLELISM}_cores"
        echo -n "$TEST_NAME: " >> "$SUMMARY_FILE"
        
        # Extract power and energy values from the JSON file
        JSON_FILE=$(find "$RESULTS_DIR" -name "gcp_cpu_profile_*.json" -exec grep -l "$TEST_NAME" {} \;)
        
        if [ -n "$JSON_FILE" ]; then
            POWER=$(grep '"program_power"' "$JSON_FILE" | cut -d: -f2 | tr -d ' ,')
            ENERGY=$(grep '"program_energy"' "$JSON_FILE" | cut -d: -f2 | tr -d ' ,')
            DURATION=$(grep '"duration"' "$JSON_FILE" | cut -d: -f2 | tr -d ' ,')
            
            echo "Power: ${POWER}W, Energy: ${ENERGY}J, Duration: ${DURATION}s" >> "$SUMMARY_FILE"
        else
            echo "No results found" >> "$SUMMARY_FILE"
        fi
    done
done

echo "" | tee -a "$LOG_FILE"
echo "Summary created: $SUMMARY_FILE" | tee -a "$LOG_FILE"
echo "All done!" | tee -a "$LOG_FILE"

# GCP: Uncomment the following line to shutdown the VM after profiling 
# sudo shutdown -h now