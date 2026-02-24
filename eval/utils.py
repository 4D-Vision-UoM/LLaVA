import yaml
import os
        
import sys
import os
import datetime

__all__ = ["load_config", "setup_global_logging"]

def load_config(filepath="config.yaml"):
    """Loads the YAML configuration file securely."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Falling back to environment variables.")
        return {}
        
    with open(filepath, 'r') as file:
        try:
            return yaml.safe_load(file) or {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return {}


class DualLogger:
    def __init__(self, filepath, original_stream):
        self.terminal = original_stream
        self.log_file = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self):
        # This ensures the text is written to the file immediately, 
        # which is vital if the script suddenly crashes!
        self.terminal.flush()
        self.log_file.flush()
# --- NEW: Methods to mimic a standard system stream ---
    def isatty(self):
        """Tells libraries not to use terminal color codes."""
        return False

    def fileno(self):
        """Returns the file descriptor of the original terminal, just in case."""
        return self.terminal.fileno()
def setup_global_logging(base_dir: str = "."):
    """
    Creates a timestamped log file and redirects all prints and errors to it.
    """
    # Generate a unique log filename based on the current time
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"eval_run_{timestamp}.log"
    log_filepath = os.path.join(base_dir, log_filename)
    
    # Hijack standard output (print statements)
    sys.stdout = DualLogger(log_filepath, sys.stdout)
    # Hijack standard error (tracebacks, exceptions, and tqdm)
    sys.stderr = DualLogger(log_filepath, sys.stderr)
    
    print(f"==================================================")
    print(f" Logging initialized: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" All output and errors will be saved to: {log_filepath}")
    print(f"==================================================\n")