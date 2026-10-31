"""Same frozen candidates/engines, independently versioned own-side reset matrix."""
import os
os.environ['V2_SELECTION_OUTPUT']='runs/v2_ownside_selection_20261031'
os.environ['V2_SELECTION_REPORT']='reports/v2_ownside_selection_20261031.md'
from scripts.v2_fresh_scheduler2_20261031 import main
if __name__=='__main__':main()
