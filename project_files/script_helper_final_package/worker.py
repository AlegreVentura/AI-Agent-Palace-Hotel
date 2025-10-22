import time
import os
from agente_faik import run_fake_agent_cycle

if __name__ == '__main__':
    # Get the absolute path to the root of the project
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    live_data_path = os.path.join(project_root, 'Experimento Cline', 'datos', 'client_live_actualizado.json')
    
    while True:
        run_fake_agent_cycle(live_data_path)
        print("--- Waiting 1 second for next cycle ---")
        time.sleep(1)
