#!/bin/bash

# Script to profile all selected tests using the CPU Monitor

# Configuration parameters
NUM_CORES_HOST=64
NUM_CORES_VM=32
TDP=280
LOOP_TIME=30  # Duration in minutes for each test to run

# List of tests to profile
TESTS=(
    "mycompress-multi"
    "mycompress-single"
    "mydecompress"
    "myfftw"
#    "myhpcc"
    "myjpeg"
#    "mynpb"
    "myspark"
    "myssl"
    "myvideoenc"
)

# Create a results directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="profile_results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Log file for overall execution
LOG_FILE="${RESULTS_DIR}/profile_all.log"

echo "===================================" | tee -a "$LOG_FILE"
echo "Starting profiling of all tests at: $(date)" | tee -a "$LOG_FILE"
echo "Results will be saved in: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "===================================" | tee -a "$LOG_FILE"

# Run each test
for TEST in "${TESTS[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "--------------------------------------" | tee -a "$LOG_FILE"
    echo "Starting test: $TEST at $(date)" | tee -a "$LOG_FILE"
    
    TEST_COMMAND="TOTAL_LOOP_TIME=$LOOP_TIME phoronix-test-suite batch-run $TEST"
    TEST_LOG="${RESULTS_DIR}/${TEST}.log"
    
    echo "Command: $TEST_COMMAND" | tee -a "$LOG_FILE"
    echo "Log file: $TEST_LOG" | tee -a "$LOG_FILE"
    
    # Run the test with CPU monitoring
    python cpu_monitor.py \
        --command "$TEST_COMMAND" \
        2>&1 | tee "$TEST_LOG"
    # python3 cpu_monitor_gcp.py \
    #     --num_cores_host "$NUM_CORES_HOST" \
    #     --num_cores_vm "$NUM_CORES_VM" \
    #     --tdp "$TDP" \
    #     --command "$TEST_COMMAND" \
    #     2>&1 | tee "$TEST_LOG"
    
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
    echo -n "$TEST: " >> "$SUMMARY_FILE"
    
    # Extract power and energy values from the JSON file
    JSON_FILE=$(find "$RESULTS_DIR" -name "gcp_cpu_profile_*.json" -exec grep -l "$TEST" {} \;)
    
    if [ -n "$JSON_FILE" ]; then
        POWER=$(grep '"program_power"' "$JSON_FILE" | cut -d: -f2 | tr -d ' ,')
        ENERGY=$(grep '"program_energy"' "$JSON_FILE" | cut -d: -f2 | tr -d ' ,')
        DURATION=$(grep '"duration"' "$JSON_FILE" | cut -d: -f2 | tr -d ' ,')
        
        echo "Power: ${POWER}W, Energy: ${ENERGY}J, Duration: ${DURATION}s" >> "$SUMMARY_FILE"
    else
        echo "No results found" >> "$SUMMARY_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "Summary created: $SUMMARY_FILE" | tee -a "$LOG_FILE"
echo "All done!" | tee -a "$LOG_FILE"
