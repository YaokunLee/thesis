import psutil
import csv
import time


def record_memory(pid):
    with open('memory_usage.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        while True:
            try:
                process = psutil.Process(pid)
                process_memory = process.memory_info().rss / 1024 / 1024  # in MB
                current_time = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime())
                writer.writerow([current_time, process_memory])
                time.sleep(5)  # Sleep for 3 minutes
            except psutil.NoSuchProcess:
                print(f"Process with PID '{pid}' not found.")
                break


if __name__ == "__main__":
    pid = int(input("请输入要检测的进程PID: "))
    record_memory(pid)
