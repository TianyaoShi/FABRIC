## CPU Monitor Scripts

The repository includes power monitoring utilities for HPC benchmarking:

- **`cpu_monitor.py`**: Direct hardware power monitoring for AMD EPYC processors using MSR (Model-Specific Register) access. Reads energy consumption counters at 15.3 micro-Joule increments to provide precise power measurements during benchmark execution. Handles counter wraparound and provides both real-time monitoring and post-execution analysis.

- **`cpu_monitor_gcp.py`**: Cloud-based power estimation for Google Cloud Platform instances. Uses CPU utilization metrics combined with power-performance models to estimate power consumption when direct hardware access is not available. Implements the power curve: P_vm = (cores_vm/cores_host) × [P_idle + (P_TDP - P_idle) × u_pkg^α].

- **`profile_all.sh`**: Automated batch profiling script that runs multiple HPC benchmark suites (compression, FFT, JPEG, SSL, video encoding, Spark) while monitoring power consumption. Orchestrates the execution of Phoronix Test Suite benchmarks with integrated power monitoring for systematic performance-energy characterization.

These tools enable precise measurement of operational environmental impacts by capturing actual power consumption during representative HPC workloads across different processor generations and deployment environments.