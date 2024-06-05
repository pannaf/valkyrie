import subprocess
import sys
import re

EXCLUDED_FILES = ["remove_traces.py"]


def remove_trace_statements(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    if not lines:
        return False  # No modifications needed for empty files

    modified = False
    with open(file_path, "w") as file:
        for line in lines:
            if re.search(r"\bpdb.set_trace\(\)", line) or re.search(r"\bipdb.set_trace\(\)", line):
                modified = True
                continue
            file.write(line)

    return modified


if __name__ == "__main__":
    # Get the list of staged files
    result = subprocess.run(["git", "diff", "--name-only", "--cached"], stdout=subprocess.PIPE)
    files = result.stdout.decode().strip().split("\n")

    modified_files = []

    # Remove traces from each staged file
    for file in files:
        if file.endswith(".py") and file not in EXCLUDED_FILES:
            if remove_trace_statements(file):
                modified_files.append(file)

    # Re-stage the modified files
    if modified_files:
        subprocess.run(["git", "add"] + modified_files)
        # Run Black on the modified files
        subprocess.run(["black"] + modified_files)
        # Re-stage the formatted files
        subprocess.run(["git", "add"] + modified_files)

    # Exit with a success code if files were modified and re-staged
    sys.exit(0 if modified_files else 1)
