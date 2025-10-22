import json
import time
from pathlib import Path

def run_fake_agent_cycle(output_path):
    """
    Simulates one cycle of an AI agent by gradually populating a JSON file.
    """
    source_path = Path(__file__).parent / "datos" / "client_familia_actualizado.json"
    
    with open(source_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)

    # Start with an empty structure
    current_data = {
        "meta": {}, "client": { "nombre": "", "apellidos": "", "email": "", "member_status": "", "telefono": "", "tiene_reservacion": False, "tipo_huesped": "", "motivo": "", "pais": "", "resort": "" },
        "apis": {}, "llamada": {}, "analisis_emociones": {}, "resumen_llamada_md": ""
    }

    def update_output():
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, indent=2, ensure_ascii=False)
        print(f"Updated {output_path} with new data.")

    print("--- Starting Fake Agent Simulation Cycle ---")

    current_data["llamada"] = {"ongoing_call": 1}
    
    # 1. Add Name
    print("Adding Name...")
    current_data["client"]["nombre"] = source_data["client"]["nombre"]
    current_data["client"]["apellidos"] = source_data["client"]["apellidos"]
    update_output()
    time.sleep(3)

    # 2. Add Status
    print("Adding Membership Status...")
    current_data["client"]["member_status"] = source_data["client"]["member_status"]
    update_output()
    time.sleep(3)
    
    # 3. Add Reason
    print("Adding Call Reason...")
    current_data["client"]["motivo"] = source_data["client"]["motivo"]
    current_data["client"]["tiene_reservacion"] = source_data["client"]["tiene_reservacion"]
    update_output()
    time.sleep(2)
    
    # 4. Add the rest of the data
    print("Adding remaining data...")
    for key, value in source_data.items():
        if isinstance(value, dict):
            if key not in current_data:
                current_data[key] = {}
            for sub_key, sub_value in value.items():
                current_data[key][sub_key] = sub_value
                update_output()
                time.sleep(1)
        else:
            current_data[key] = value
            update_output()
            time.sleep(1)

    current_data["llamada"]["ongoing_call"] = 0
    update_output()

    print("--- Fake Agent Simulation Cycle Complete ---")

if __name__ == "__main__":
    run_fake_agent_cycle("datos/client_data.json")
