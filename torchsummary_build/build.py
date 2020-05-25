import os
import re
import shutil
import sys

import f2format
from strip_hints import strip_file_to_string


def remove_duplicate_files(destination, filename):
    # Remove file if it already exists.
    full_dest_path = os.path.join(destination, filename)
    if filename in os.listdir(destination):
        if os.path.isdir(full_dest_path):
            shutil.rmtree(full_dest_path)
        else:
            os.remove(full_dest_path)


def copy_file_or_folder(source, destination, filename):
    full_src_path = os.path.join(source, filename)
    full_dest_path = os.path.join(destination, filename)
    if os.path.isdir(full_src_path):
        shutil.copytree(full_src_path, full_dest_path)
    else:
        shutil.copy(full_src_path, full_dest_path)


def copy_src(source, destination, filename):
    full_src_path = os.path.join(source, filename)
    full_dest_path = os.path.join(destination, filename)
    with open(full_dest_path, "w") as f:
        code_string = strip_file_to_string(full_src_path, to_empty=True, strip_nl=True)

        code_string = re.sub(r"from typing import .*\n", "", code_string)
        code_string = re.sub(r"\n.*INPUT_SIZE_TYPE = .*\n", "", code_string)
        code_string = re.sub(r"DETECTED_OUTPUT_TYPES = .*\n", "", code_string)
        code_string = re.sub(r" *,", ",", code_string)
        code_string = re.sub(r" +=", " =", code_string)
        code_string = re.sub(r",\n\s*", ", ", code_string)
        code_string = re.sub(r"\(\n\s*", "(", code_string)
        code_string = re.sub(r",?\s*\)\s*:", "):", code_string)

        code_string = code_string.replace("CORRECTED_INPUT_SIZE_TYPE, ", "")
        code_string = code_string.replace("from __future__ import annotations\n", "")

        code_string = f2format.convert(code_string)
        f.write(code_string)


def create_project_folder():
    source = "../torchsummary"
    destination = "./torchsummary"

    # Create destination directory if it doesn't exist
    if not os.path.isdir(destination):
        os.makedirs(destination)

    for filename in os.listdir(source):
        if filename.endswith(".py"):
            remove_duplicate_files(destination, filename)
            copy_src(source, destination, filename)


if __name__ == "__main__":
    if sys.version_info < (3, 7):
        sys.stdout.write("Python " + sys.version)
        sys.stdout.write("\n\nRequires Python 3.7+ to work!\n\n")
        sys.exit()
    else:
        create_project_folder()
