import os
import re
import shutil
import string

# import astunparse
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

    # with open(full_src_path) as src_f:
    #     with open(full_dest_path, "w") as dst_f:
    #         # parse the source code into an AST
    #         parsed_source = ast.parse(src_f.read())
    #         # remove all type annotations, function return type definitions
    #         # and import statements from 'typing'
    #         transformed = TypeHintRemover().visit(parsed_source)
    #         # convert the AST back to source code
    #         code_string = astunparse.unparse(transformed)
    #         dst_f.write(code_string)


def fill_template_files(destination, config):
    # Fill in template files with entries in config.
    for root, _, files in os.walk(destination):
        if (
            "data" in root or "checkpoints" in root or "cache" in root
        ):  # TODO use .gitignore instead
            continue
        for filename in files:
            if "_temp.py" in filename:
                result = ""
                full_src_path = os.path.join(root, filename)
                with open(full_src_path) as in_file:
                    contents = string.Template(in_file.read())
                    result = contents.substitute(config["substitutions"])

                new_dest_path = full_src_path.replace("_temp", "")
                with open(new_dest_path, "w") as out_file:
                    out_file.write(result)
                os.remove(full_src_path)


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


def main():
    create_project_folder()


if __name__ == "__main__":
    main()
