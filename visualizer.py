import csv
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import os

# Define colors for agents
COLORS = {
    "Human": "red",
    "Zombie": "purple"
}

def plot_agents(args):
    tick, data = args
    print(f"[Process {os.getpid()}] Generating image for tick {tick}...")
    plt.figure(figsize=(10, 10))
    
    for agent_type, color in COLORS.items():
        x_vals = [x for at, _, x, y, _ in data if at == agent_type]
        y_vals = [y for at, _, x, y, _ in data if at == agent_type]
        plt.scatter(x_vals, y_vals, color=color, label=agent_type)

    plt.title(f"Tick {tick}")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"tick_{tick}.png")
    plt.close()
    print(f"[Process {os.getpid()}] Image for tick {tick} saved as tick_{tick}.png")

def main():
    with open("agent_data.csv", "r") as file:
        reader = csv.reader(file)
        next(reader)

        data = []
        current_tick = 1
        all_ticks_data = []

        for row in reader:
            tick, agent_type, agent_id, x, y = row[:5]
            status = row[5] if len(row) > 5 else None
            tick = int(tick)
            x, y = float(x), float(y)
            if tick != current_tick:
                all_ticks_data.append((current_tick, data))
                data = []
                current_tick = tick
            data.append((agent_type, int(agent_id), x, y, status))
        all_ticks_data.append((current_tick, data))

        # Use multiprocessing with a number of processes equal to the number of available CPU cores
        with Pool(processes=cpu_count()) as p:
            p.map(plot_agents, all_ticks_data)

if __name__ == "__main__":
    main()
