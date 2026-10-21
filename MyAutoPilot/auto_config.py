import re
import subprocess
from pathlib import Path

# Current ETS2 sensitivity
steering_sensitivity = 3.0
relative_steering_sensitivity = 0.5

def update_controls(controls_text):
    if not controls_text.lstrip("\ufeff").startswith("SiiNunit"):
        raise ValueError("Expected a text controls.sii file")

    # Wheel mode , steering deadzone and sensitivity
    constants = {
        "c_relatsteer": 0.0 ,
        "c_mousesteer": 0.0 ,
        "c_steer_dz": 0.0 ,
        "c_asteersens": steering_sensitivity ,
        "c_rsteersens": relative_steering_sensitivity
    }
    for name , value in constants.items():
        pattern = r'(?m)^([ \t]*config_lines\[\d+\]: "constant ' + name + r' )[^"\r\n]+(")(?=\r?$)'
        controls_text , count = re.subn(pattern , lambda match: match[1] + f"{value:.6f}" + match[2] , controls_text)
        if count != 1:
            raise ValueError("Expected one constant: " + name)

    # Release lane change keys from driving only
    for name in ("dsteerleft" , "dsteerright"):
        pattern = r'(?m)^[ \t]*config_lines\[\d+\]: "mix ' + name + r' `([^`\r\n]*)`"(?=\r?$)'
        matches = list(re.finditer(pattern , controls_text))
        if len(matches) != 1:
            raise ValueError("Expected one mix: " + name)
        match = matches[0]
        binding = re.sub(r'(?<![\w.])keyboard\.(?:larrow|rarrow)(?:\?0)?(?![\w.?])' , "unbound?0" , match[1])
        controls_text = controls_text[:match.start(1)] + binding + controls_text[match.end(1):]

    return controls_text

def auto_config(controls_path):
    controls_path = Path(controls_path)

    # Validate before touching the file
    original = controls_path.read_bytes()
    updated = update_controls(original.decode("utf-8")).encode("utf-8")
    if updated == original:
        print("ETS2 controls already configured:" , controls_path)
        return False

    # ETS2 saves controls on exit
    process = subprocess.run(
        ["tasklist" , "/FI" , "IMAGENAME eq eurotrucks2.exe" , "/FO" , "CSV" , "/NH"] ,
        capture_output = True , check = True , creationflags = subprocess.CREATE_NO_WINDOW
    )
    if b'"eurotrucks2.exe"' in process.stdout.lower():
        raise RuntimeError("Close ETS2 before applying controls")

    # Keep the original backup on repeated runs
    backup_path = controls_path.with_name("controls.sii.bak")
    if not backup_path.exists():
        backup_path.write_bytes(original)
    controls_path.write_bytes(updated)
    print("ETS2 controls configured:" , controls_path)
    print("Original controls backup:" , backup_path)
    return True
