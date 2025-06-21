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
The CPU energy reading for AMD chips
AMD EPYC 7443 24-Core Processor
"""
ENERGY_INCREMENT = 15.3e-6 # energy status unit is in 15.3 micro-Joules increment.
MAX_COUNTER_VALUE = 2**32 # maximum value for 32 bit
TIME_OUT=86400 # 2 hours timeout for the command


class CPUMonitor(object):
    def __init__(self,
                 project_name='monitor',
                 cpu_model='AMD EPYC 7443 24-Core Processor',
                 os_version='Linux-5.15.0-128-generic-x86_64-with-glibc2.35',
                 region='IN,USA',
                 carbon_intensity=369,  # US average
                 cpu_id=None,
                 msr_address: str="0xc001029B",
                 monitor_interval: float = 0.5,
                 password=None):
        self.project_name = project_name
        self.cpu_id = cpu_id
        self.msr_address = msr_address
        self.monitor_interval = monitor_interval
        self.password = password

        # Signal to stop monitoring
        self.stop_event = multiprocessing.Event()
        
        # Process for monitoring
        self.process = None

        # Instead of a queue, use a file for output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = f"{self.project_name}_cpu{self.cpu_id}_{timestamp}.csv"
        
        # Initialize the CSV file with headers
        with open(self.output_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'energy_counter', 'energy_joules', 'cumulative_energy'])
        
        # Store precise start and end times
        self.start_time = None
        self.end_time = None

        # Fix the tuple issue - remove trailing commas
        self.cpu_model = cpu_model
        self.os_version = os_version
        self.region = region
        self.carbon_intensity = carbon_intensity

    def _read_energy_consumption(self):
        try:
            if self.password:
                # Using shell=True because we pipe in the password.
                command = f"echo {self.password} | sudo -S rdmsr -p {str(self.cpu_id)} {self.msr_address}"
                result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            else:
                # Without a password, assume all CPUs read the same energy consumption.
                command = ["sudo", "rdmsr", self.msr_address]
                result = subprocess.run(command, capture_output=True, text=True, check=True)

            # Extract the hexadecimal output and convert to decimal.
            hex_value = result.stdout.strip()
            decimal_value = int(hex_value, 16)
            return decimal_value
        except subprocess.CalledProcessError as e:
            print(f"Error executing rdmsr: {e}", file=sys.stderr)
            return None
        except FileNotFoundError:
            print("Error: 'rdmsr' command not found. Is msr-tools installed?", file=sys.stderr)
            return None
        except ValueError as e:
            print(f"Error converting value to decimal: {e}", file=sys.stderr)
            return None

    def _monitor_cpu(self):
        """Monitor CPU energy consumption and write results directly to file."""
        last_timestamp = time.time()
        last_energy_reading = self._read_energy_consumption()
        cumulative_energy = 0.0  # Total energy in Joules
        next_sample_time = last_timestamp

        if last_energy_reading is None:
            print(f"Error: Could not read energy register for CPU {self.cpu_id}", file=sys.stderr)
            return

        # Open file in append mode for continuous writing
        with open(self.output_file, 'a') as f:
            writer = csv.writer(f)
            
            # Write initial reading
            writer.writerow([last_timestamp, last_energy_reading, 0, 0])
            f.flush()
            
            while not self.stop_event.is_set():
                current_time = time.time()
                
                # Time to take a sample?
                if current_time >= next_sample_time:
                    try:
                        # Read the energy register
                        current_energy_reading = self._read_energy_consumption()
                        
                        if current_energy_reading is not None:
                            # Handle counter wraparound
                            if current_energy_reading >= last_energy_reading:
                                # No reset, straightforward increment
                                energy_delta = current_energy_reading - last_energy_reading
                            else:
                                # Reset occurred, add wraparound increment
                                energy_delta = (MAX_COUNTER_VALUE - last_energy_reading) + current_energy_reading
                            
                            # Convert to Joules
                            energy_joules = energy_delta * ENERGY_INCREMENT
                            cumulative_energy += energy_joules
                            
                            # Write to CSV
                            writer.writerow([current_time, current_energy_reading, energy_joules, cumulative_energy])
                            f.flush()  # Ensure data is written immediately
                            
                            # Update state for next iteration
                            last_timestamp = current_time
                            last_energy_reading = current_energy_reading
                        
                        # Schedule next sample with precise timing
                        next_sample_time += self.monitor_interval
                        
                        # Reset if we're falling behind
                        if current_time > next_sample_time + self.monitor_interval:
                            next_sample_time = current_time + self.monitor_interval
                            print(f"Warning: Monitor for CPU {self.cpu_id} fell behind schedule, resetting timing", file=sys.stderr)
                            
                    except Exception as e:
                        print(f"Error in CPU monitoring for CPU {self.cpu_id}: {e}", file=sys.stderr)
                
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
        print(f"Monitoring started for CPU {self.cpu_id}. Output file: {self.output_file}")

    def stop(self):
        """Signal the monitor to stop and wait for the process to finish."""
        self.end_time = time.time()
        if self.process is not None and self.process.is_alive():
            self.stop_event.set()
            self.process.join(timeout=2)  # Wait up to 2 seconds
            if self.process.is_alive():
                print(f"CPU monitor on CPU {self.cpu_id} did not terminate gracefully, terminating...")
                self.process.terminate()
                self.process.join(timeout=1)
                if self.process.is_alive():
                    print(f"CPU monitor on CPU {self.cpu_id} still alive, killing...")
                    self.process.kill()
        print(f"Monitoring stopped for CPU {self.cpu_id}. Duration: {self.end_time - self.start_time:.2f} seconds")

    def close(self):
        """Clean up resources used by the monitor."""
        self.stop()  # Make sure the process is stopped

    def get_summary(self):
        """
        Calculate summary statistics from the monitoring data.
        
        Returns:
            Dictionary containing power, energy and timing information
        """
        try:
            if not os.path.exists(self.output_file):
                return {"error": f"No monitoring data found for CPU {self.cpu_id}"}
                
            if not self.start_time or not self.end_time:
                return {"error": "Monitor timing information unavailable"}
                
            # Read the CSV file and calculate statistics
            measurements = []
            with open(self.output_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 4:
                        timestamp, energy_counter, energy_joules, cumulative_energy = map(float, row)
                        measurements.append((timestamp, energy_counter, energy_joules, cumulative_energy))
            
            if not measurements:
                return {
                    "cpu_id": self.cpu_id,
                    "samples_count": 0,
                    "avg_power": 0,
                    "total_energy": 0,
                    "monitor_duration": self.end_time - self.start_time
                }
            
            # Calculate statistics
            first_timestamp = measurements[0][0]
            last_timestamp = measurements[-1][0]
            monitor_duration = last_timestamp - first_timestamp
            
            # Take the final cumulative energy value
            total_energy = measurements[-1][3] if measurements else 0
            
            # Calculate average power
            avg_power = total_energy / monitor_duration if monitor_duration > 0 else 0
            
            # Calculate total energy based on average power and wall-clock time
            wall_clock_duration = self.end_time - self.start_time
            wall_clock_energy = avg_power * wall_clock_duration
            
            return {
                "cpu_id": self.cpu_id,
                "samples_count": len(measurements),
                "avg_power": avg_power,
                "total_energy": total_energy,
                "wall_clock_energy": wall_clock_energy,
                "monitor_duration": monitor_duration,
                "wall_clock_duration": wall_clock_duration,
                "coverage_percentage": (monitor_duration / wall_clock_duration) * 100 if wall_clock_duration > 0 else 0,
                "data_file": self.output_file
            }
            
        except Exception as e:
            print(f"Error calculating summary for CPU {self.cpu_id}: {e}", file=sys.stderr)
            return {"error": str(e), "cpu_id": self.cpu_id}
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU monitor")
    parser.add_argument('--cpu_ids', type=str,
                        default='0',  # Default CPUs
                        help='Comma-separated list of CPU IDs to use (e.g., 0,1,2)')
    parser.add_argument('--password', type=str,
                        default=None,   # !!! insecure, internal test only
                        help='Sudo password')
    parser.add_argument('--command', type=str,
                        help='CLI command to profile')

    args = parser.parse_args()
    import signal
    shutdown_requested = False

    cpu_ids = list(map(int, args.cpu_ids.split(',')))
    
    def signal_handler(sig, frame):
        global shutdown_requested
        print("Signal received, initiating graceful shutdown...")
        shutdown_requested = True
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # First measure idle power for 30 seconds
    print("Measuring idle power for 30 seconds...")
    idle_monitors = [CPUMonitor(project_name='idle_power', 
                               cpu_id=cpu_id, 
                               msr_address="0xc001029B", 
                               monitor_interval=0.1, 
                               password=args.password) 
                     for cpu_id in cpu_ids]
    
    for cpu_monitor in idle_monitors:
        cpu_monitor.start()
    
    time.sleep(30)  # Profile idle power for 30 seconds

    idle_powers = []
    for cpu_monitor in idle_monitors:
        cpu_monitor.stop()
        summary = cpu_monitor.get_summary()
        idle_powers.append(summary.get('avg_power', 0))
        cpu_monitor.close()

    # Reset monitors for program profiling
    program_monitors = [CPUMonitor(project_name='program_power', 
                                  cpu_id=cpu_id, 
                                  msr_address="0xc001029B", 
                                  monitor_interval=1, 
                                  password=args.password) 
                        for cpu_id in cpu_ids]

    # Start monitoring and run the command
    print("Starting CPU monitoring and profiling...")
    start_time = time.time()
    for cpu_monitor in program_monitors:
        cpu_monitor.start()

    if args.command:
        try:
            print(f"Running command: {args.command}")
            process = subprocess.Popen(args.command, shell=True)
            
            # Monitor the process while it's running
            while process.poll() is None:
                time.sleep(0.5)  # Check status more frequently
                
                # Ensure all monitoring processes are still alive
                for i, cpu_monitor in enumerate(program_monitors):
                    if cpu_monitor.process is None or not cpu_monitor.process.is_alive():
                        print(f"Warning: Monitor process for CPU {cpu_ids[i]} died, restarting...")
                        cpu_monitor.stop()
                        cpu_monitor.start()
                
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
            shutdown_requested = True
        finally:
            # Ensure monitors are stopped even if an exception occurs
            print("Stopping CPU monitors...")
            for cpu_monitor in program_monitors:
                cpu_monitor.stop()

    end_time = time.time()

    # Get summaries from each monitor
    program_summaries = []
    for cpu_monitor in program_monitors:
        summary = cpu_monitor.get_summary()
        program_summaries.append(summary)
        cpu_monitor.close()

    # Extract relevant values for results
    program_powers = [summary.get('avg_power', 0) for summary in program_summaries]
    program_energies = [summary.get('total_energy', 0) for summary in program_summaries]
    total_power = sum(program_powers)
    total_energy = sum(program_energies)
    
    # Prepare results
    results = {
        "start_timestamp": start_time,
        "end_timestamp": end_time,
        "command": args.command if args.command else "No command specified",
        "cpu_ids": cpu_ids,
        "idle_powers": idle_powers,
        "program_powers": program_powers,
        "program_energies": program_energies,
        "total_power": total_power,
        "total_energy": total_energy,
        "duration": end_time - start_time,
        "monitor_details": program_summaries
    }

    # Save results to JSON file
    timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(start_time))
    filename = f"cpu_profile_{timestamp_str}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {filename}")
    print("\nExecution Summary:")
    print(f"Command: {args.command if args.command else 'No command specified'}")
    print(f"Duration: {results['duration']:.2f} seconds")
    print(f"Total power across all CPUs: {total_power:.2f} Watts")
    print(f"Total energy: {total_energy:.2f} Joules")
    
    # Print individual CPU results
    print("\nPer-CPU Results:")
    for i, summary in enumerate(program_summaries):
        cpu_id = cpu_ids[i]
        print(f"CPU {cpu_id}: {summary.get('avg_power', 0):.2f} W, {summary.get('total_energy', 0):.2f} J, {summary.get('coverage_percentage', 0):.1f}% coverage")
        print(f"  Data file: {summary.get('data_file', 'unknown')}")
