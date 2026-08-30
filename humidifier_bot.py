import logging
import logging.handlers
import os
import subprocess
import time
from datetime import datetime

# --- Configuration ---
# Name of the Shortcuts you created on your Mac
SHORTCUT_ON_NAME = "PH On"
SHORTCUT_OFF_NAME = "PH Off"

# Cycle Configuration (in seconds)
ON_DURATION = 780   # 13 minutes
OFF_DURATION = 1200  # 20 minutes
# A full cycle is: ON -> Wait -> OFF -> Wait -> ON -> Wait -> OFF
# Total Active Time: 30 minutes

# Scheduling
START_HOUR = 7
END_HOUR = 20  # 8 PM

# Interval between 30-minute cycles (in seconds)
# Set to 0 to loop immediately (continuous).
# Set to 1800 (30 mins) if you want a 30m gap between cycles.
CYCLE_GAP = 0 

# Set to True to print actions instead of running them
TEST_MODE = False

# Setup logging — file + console
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "humidifier.log")

logger = logging.getLogger("humidifier")
logger.setLevel(logging.INFO)

_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Rotating file handler: 1 MB max, keep 3 backups
_fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3)
_fh.setFormatter(_fmt)
logger.addHandler(_fh)

# Console handler (so it still prints if run interactively)
_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

def run_shortcut(shortcut_name):
    """Runs a macOS Shortcut via command line."""
    if TEST_MODE:
        logger.info(f"[TEST] Would run shortcut: '{shortcut_name}'")
        return True
    
    try:
        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Running '{shortcut_name}'...")
        # 'shortcuts run' is the native macOS command
        subprocess.run(["shortcuts", "run", shortcut_name], check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running shortcut: {e}")
        return False
    except FileNotFoundError:
        logger.error("Error: 'shortcuts' command not found. Are you on macOS Monterey or later?")
        return False

def run_cycle():
    """Runs the specific 30-minute cycle requested."""
    logger.info("--- Starting 30-Minute Cycle ---")
    
    # 0-15 mins: ON
    run_shortcut(SHORTCUT_ON_NAME)
    time.sleep(ON_DURATION)
    
    # 15-30 mins: OFF
    run_shortcut(SHORTCUT_OFF_NAME)
    time.sleep(OFF_DURATION)
    
    logger.info("--- Cycle Complete ---")

def main():
    logger.info("Starting Humidifier Bot...")
    if TEST_MODE:
        logger.warning("!! TEST MODE ENABLED - No actual shortcuts will be run !!")
    
    while True:
        now = datetime.now()
        current_hour = now.hour
        
        # Check if within schedule (7 AM to 8 PM)
        # We stop at 8:00 PM (hour 20)
        if START_HOUR <= current_hour < END_HOUR:
            run_cycle()
            
            if CYCLE_GAP > 0:
                logger.info(f"Waiting {CYCLE_GAP/60:.1f} minutes before next cycle...")
                time.sleep(CYCLE_GAP)
        else:
            # If outside schedule, sleep a bit to avoid busy looping
            # E.g., check every 5 minutes until 7 AM
            logger.info(f"[{now.strftime('%H:%M')}] Outside hours ({START_HOUR}-{END_HOUR}). Sleeping...")
            time.sleep(300)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nExiting Bot.")