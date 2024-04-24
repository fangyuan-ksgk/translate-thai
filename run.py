from translate import *
# process_file("")
import glob
import os
file_names = glob.glob("data/*.csv")
file_name = file_names[1]

process_file(file_name, api_key = os.environ['ANTHROPIC_API_KEY'])

