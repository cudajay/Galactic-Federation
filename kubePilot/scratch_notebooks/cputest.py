import multiprocessing
import time
import csv
import sys

# Function to simulate CPU usage
def simulate_cpu_usage(percentage, duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        pass

if __name__ == '__main__':
    try:
        # Increment step for CPU usage percentage
        step = 10

        print("Press Ctrl+C to stop the stress test.")
        with open('cpu_power_usage.csv', 'w', newline='') as csvfile:
            fieldnames = ['CPU Usage (%)', 'Power Consumption (W)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                try:
                    current_usage = 0

                    while current_usage <= 100:
                        target_usage_per_process = (current_usage / 100) * 100
                        processes = []

                        # Start the processes
                        for i in range(multiprocessing.cpu_count()):
                            p = multiprocessing.Process(target=simulate_cpu_usage, args=(target_usage_per_process, 10))
                            p.start()
                            processes.append(p)

                        # Join the processes
                        for p in processes:
                            p.join()

                        # Prompt the user to enter measured power consumption
                        measured_power = float(input(f"Enter measured power consumption for {current_usage}% CPU usage (in watts): "))
                        if measured_power <= 0:
                            print("Please enter a positive value for measured power consumption.")
                            continue

                        # Write data to CSV
                        writer.writerow({'CPU Usage (%)': current_usage, 'Power Consumption (W)': measured_power})

                        print(f"CPU usage: {current_usage}%, Power consumption: {measured_power:.2f} W")
                        current_usage += step
                        time.sleep(2)  # Hold the usage for a few seconds before incrementing

                except KeyboardInterrupt:
                    print("\nTest interrupted by user.")
                    sys.exit()

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
