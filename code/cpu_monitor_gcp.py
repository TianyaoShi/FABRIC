#!/usr/bin/env python3
import multiprocessing
import time
import subprocess
import sys
import argparse
import os
import json
import psutil
import signal
import csv
from datetime import datetime

"""
GCP CPU energy monitoring using CPU utilization as a proxy
Using the power-util curve:
u_pkg = (num_cores_vm / num_cores_host) * u_vm
P_vm = (num_cores_vm / num_cores_host) * [P(idle) + [P(TDP) - P(idle)] * u_pkg^α]

where:
- u_vm is the CPU utilization of VM (percentage between 0-1)
- num_cores_vm is the number of cores in the VM
- num_cores_host is the number of cores in the host machine
- P(idle) is assumed to be 30% of TDP
- α is set to 1.15
"""

TIME_OUT = 3600  # 1 hour

class GCPCPUMonitor:
    def __init__(self,
                 project_name='monitor',
                 num_cores_vm=None,  # Number of vCPUs in the VM
                 num_cores_host=None,  # Number of cores in host
                 tdp=None,  # Thermal Design Power in Watts
                 alpha=1.15,  # Power model exponent
                 idle_power_ratio=0.3,  # P(idle) as a ratio of TDP
                 monitor_interval=0.5):
        """Initialize the CPU monitor."""
        self.project_name = project_name
        
        # If num_cores_vm is not provided, detect it
        self.num_cores_vm = num_cores_vm if num_cores_vm is not None else psutil.cpu_count(logical=True)
        self.num_cores_host = num_cores_host
        self.tdp = tdp
        self.alpha = alpha
        self.monitor_interval = monitor_interval
        self.idle_power_ratio = idle_power_ratio
        
        # Create a unique output file for this monitoring session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = f"{self.project_name}_{timestamp}.csv"
        
        # Initialize the CSV file with headers
        with open(self.output_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'cpu_util', 'power', 'energy'])
        
        # Signal to stop monitoring
        self.stop_event = multiprocessing.Event()
        
        # Monitor process
        self.process = None
        
        # Tracking timestamps
        self.start_time = None
        self.end_time = None
        
        # Validate required parameters
        if self.num_cores_host is None or self.tdp is None:
            raise ValueError("num_cores_host and tdp are required parameters")

    def _calculate_power(self, cpu_util):
        """
        Calculate power consumption using the power-util curve.
        
        Args:
            cpu_util: CPU utilization (0-1)
            
        Returns:
            Estimated power consumption in Watts
        """
        # Calculate u_pkg = (num_cores_vm / num_cores_host) * u_vm
        u_pkg = (self.num_cores_vm / self.num_cores_host) * cpu_util
        
        # Calculate P(idle)
        p_idle = self.tdp * self.idle_power_ratio
        
        # Calculate P_vm = (num_cores_vm / num_cores_host) * [P(idle) + [P(TDP) - P(idle)] * u_pkg^α]
        p_vm = (self.num_cores_vm / self.num_cores_host) * (p_idle + (self.tdp - p_idle) * (u_pkg ** self.alpha))
        
        return p_vm

    def _get_cpu_utilization(self):
        """Get current CPU utilization as a value between 0 and 1."""
        return psutil.cpu_percent(interval=None) / 100.0

    def _monitor_cpu(self):
        """Monitor CPU and write results directly to file."""
        # Initialize CPU monitoring 
        _ = self._get_cpu_utilization()  # First call initializes cpu monitoring
        
        last_timestamp = time.time()
        last_energy = 0.0  # Accumulated energy in Joules
        next_sample_time = last_timestamp
        
        # Open file in append mode for continuous writing
        with open(self.output_file, 'a') as f:
            writer = csv.writer(f)
            
            while not self.stop_event.is_set():
                current_time = time.time()
                
                # Time to take a sample?
                if current_time >= next_sample_time:
                    try:
                        # Get CPU utilization and calculate power
                        cpu_util = self._get_cpu_utilization()
                        power = self._calculate_power(cpu_util)
                        
                        # Calculate energy increment (power * time)
                        delta_time = current_time - last_timestamp
                        energy_increment = power * delta_time
                        current_energy = last_energy + energy_increment
                        
                        # Write data directly to file
                        writer.writerow([current_time, cpu_util, power, current_energy])
                        f.flush()  # Ensure data is written immediately
                        
                        # Update state for next iteration
                        last_timestamp = current_time
                        last_energy = current_energy
                        
                        # Schedule next sample with precise timing
                        next_sample_time += self.monitor_interval
                        
                        # Reset if we're falling behind
                        if current_time > next_sample_time + self.monitor_interval:
                            next_sample_time = current_time + self.monitor_interval
                            print(f"Warning: Monitor fell behind schedule, resetting timing", file=sys.stderr)
                            
                    except Exception as e:
                        print(f"Error in CPU monitoring: {e}", file=sys.stderr)
                        
                # Sleep just enough to hit next sampling time
                sleep_time = max(0.001, min(next_sample_time - time.time(), self.monitor_interval/2))
                time.sleep(sleep_time)

    def start(self):
        """Start the monitor in a separate process."""
        self.stop_event.clear()
        self.process = multiprocessing.Process(target=self._monitor_cpu)
        self.process.daemon = True  # Ensure process terminates when main program exits
        self.process.start()
        self.start_time = time.time()
        print(f"Monitoring started. Output file: {self.output_file}")

    def stop(self):
        """Signal the monitor to stop and wait for the process to finish."""
        self.end_time = time.time()
        if self.process is not None and self.process.is_alive():
            self.stop_event.set()
            self.process.join(timeout=2)
            if self.process.is_alive():
                print("CPU monitor did not terminate gracefully, terminating...")
                self.process.terminate()
                self.process.join(timeout=1)
                if self.process.is_alive():
                    print("CPU monitor still alive, killing...")
                    self.process.kill()
        print(f"Monitoring stopped. Duration: {self.end_time - self.start_time:.2f} seconds")

    def get_summary(self):
        """
        Calculate summary statistics from the monitoring data.
        
        Returns:
            Dictionary containing power, energy and timing information
        """
        try:
            if not os.path.exists(self.output_file):
                return {"error": "No monitoring data found"}
                
            if not self.start_time or not self.end_time:
                return {"error": "Monitor timing information unavailable"}
                
            # Read the CSV file and calculate statistics
            measurements = []
            with open(self.output_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    timestamp, cpu_util, power, energy = map(float, row)
                    measurements.append((timestamp, cpu_util, power, energy))
            
            if not measurements:
                return {
                    "samples_count": 0,
                    "avg_power": 0,
                    "total_energy": 0,
                    "monitor_duration": self.end_time - self.start_time
                }
            
            # Calculate statistics
            first_timestamp = measurements[0][0]
            last_timestamp = measurements[-1][0]
            monitor_duration = last_timestamp - first_timestamp
            
            # Calculate average power from all samples
            total_power = sum(m[2] for m in measurements)
            avg_power = total_power / len(measurements) if measurements else 0
            
            # Take the final cumulative energy value
            total_energy = measurements[-1][3] if measurements else 0
            
            # Calculate total energy based on average power and wall-clock time
            wall_clock_duration = self.end_time - self.start_time
            wall_clock_energy = avg_power * wall_clock_duration
            
            return {
                "samples_count": len(measurements),
                "avg_power": avg_power,
                "total_energy": total_energy,
                "wall_clock_energy": wall_clock_energy,
                "monitor_duration": monitor_duration,
                "wall_clock_duration": wall_clock_duration,
                "coverage_percentage": (monitor_duration / wall_clock_duration) * 100 if wall_clock_duration > 0 else 0
            }
            
        except Exception as e:
            print(f"Error calculating summary: {e}", file=sys.stderr)
            return {"error": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCP CPU Power Monitor")
    parser.add_argument('--num_cores_host', type=int, required=True,
                        help='Number of physical cores in the host machine')
    parser.add_argument('--num_cores_vm', type=int, default=None,
                        help='Number of vCPUs in the VM (defaults to detected count)')
    parser.add_argument('--tdp', type=float, required=True,
                        help='Thermal Design Power (TDP) in Watts')
    parser.add_argument('--alpha', type=float, default=1.15,
                        help='Power model exponent (default: 1.15)')
    parser.add_argument('--idle_power_ratio', type=float, default=0.3,
                        help='Power at idle as a ratio of TDP (default: 0.3)')
    parser.add_argument('--command', type=str,
                        help='CLI command to profile')

    args = parser.parse_args()
    shutdown_requested = False

    def signal_handler(sig, frame):
        global shutdown_requested
        print("Signal received, initiating graceful shutdown...")
        shutdown_requested = True
        
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # First measure idle power for 30 seconds
    print("Measuring idle power for 30 seconds...")
    idle_monitor = GCPCPUMonitor(
        project_name='idle_power',
        num_cores_vm=args.num_cores_vm,
        num_cores_host=args.num_cores_host,
        tdp=args.tdp,
        alpha=args.alpha,
        idle_power_ratio=args.idle_power_ratio,
        monitor_interval=0.1
    )
    
    idle_monitor.start()
    time.sleep(30)  # Profile idle power for 30 seconds
    idle_monitor.stop()
    
    idle_summary = idle_monitor.get_summary()
    idle_power = idle_summary.get('avg_power', 0)
    print(f"Average idle power: {idle_power:.2f} Watts")

    # Now monitor the command execution
    if args.command:
        program_monitor = GCPCPUMonitor(
            project_name='program_power',
            num_cores_vm=args.num_cores_vm,
            num_cores_host=args.num_cores_host,
            tdp=args.tdp,
            alpha=args.alpha,
            idle_power_ratio=args.idle_power_ratio,
            monitor_interval=0.1
        )
        
        start_time = time.time()
        program_monitor.start()
        
        try:
            print(f"Running command: {args.command}")
            process = subprocess.Popen(args.command, shell=True)
            
            # Monitor the process while it's running
            while process.poll() is None:
                time.sleep(0.5)  # Check status more frequently
                
                # Ensure the monitoring process is still alive
                if program_monitor.process is None or not program_monitor.process.is_alive():
                    print("Warning: Monitor process died, restarting...")
                    program_monitor.stop()
                    program_monitor.start()
                
                # Check for timeout or shutdown request
                if time.time() - start_time > TIME_OUT:
                    print("Timeout reached, terminating process.")
                    process.terminate()
                    break
                    
                if shutdown_requested:
                    print("Shutdown requested, terminating process.")
                    process.terminate()
                    break
                    
            # Wait for process to complete
            process.wait()
            
        except KeyboardInterrupt:
            print("Process interrupted by user.")
            if process.poll() is None:
                process.terminate()
                
        except Exception as e:
            print(f"Command execution failed: {e}")
            
        finally:
            # Always stop monitoring
            program_monitor.stop()
            end_time = time.time()
            
        # Get the results
        program_summary = program_monitor.get_summary()
        
        # Prepare the final results
        results = {
            "start_timestamp": start_time,
            "end_timestamp": end_time,
            "command": args.command,
            "vm_cores": program_monitor.num_cores_vm,
            "host_cores": program_monitor.num_cores_host,
            "tdp": program_monitor.tdp,
            "alpha": program_monitor.alpha,
            "idle_power": idle_power,
            "program_power": program_summary.get('avg_power', 0),
            "program_energy": program_summary.get('wall_clock_energy', 0),
            "duration": end_time - start_time,
            "monitor_coverage": program_summary.get('coverage_percentage', 0),
            "samples_collected": program_summary.get('samples_count', 0),
            "data_file": program_monitor.output_file
        }
        
        # Print summary information
        print("\nExecution Summary:")
        print(f"Command: {args.command}")
        print(f"Duration: {results['duration']:.2f} seconds")
        print(f"Average power: {results['program_power']:.2f} Watts")
        print(f"Total energy: {results['program_energy']:.2f} Joules")
        print(f"Monitoring coverage: {results['monitor_coverage']:.1f}%")
        print(f"Samples collected: {results['samples_collected']}")
        
        # Save results to JSON
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"gcp_cpu_profile_{timestamp_str}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"Results saved to: {results_file}")
        print(f"Raw monitoring data: {program_monitor.output_file}")
    else:
        print("No command specified. Only idle power was measured.")
