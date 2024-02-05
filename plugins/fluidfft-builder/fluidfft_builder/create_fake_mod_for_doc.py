from importlib import resources
from pathlib import Path

here = Path(__file__).absolute().parent


def get_docstring(lines, indent="    "):
    doc = None

    # find start doc
    for i, line in enumerate(lines):
        # search for end of code
        if len(line) - len(line.lstrip(" ")) < len(indent):
            break

        if line.startswith(indent + '"""'):
            lines_to_be_parsed = lines[i + 1 :]
            doc = line
            break

    if doc is None or len(doc.split('"""')) > 2:
        return doc

    # find end doc
    for i, line in enumerate(lines_to_be_parsed):
        doc += "\n" + line
        if '"""' in line:
            break

    return doc


strings_to_be_deleted = [
    " int ",
    " view3df_t ",
    " view3dc_t ",
    " view2df_t ",
    " view2dc_t ",
    "const ",
]


def get_function_code(lines):
    signature = lines[0].replace(" cpdef ", " def ")

    if not signature.endswith("):"):
        for i, line in enumerate(lines[1:]):
            signature += " " + line.strip()
            if line.endswith("):"):
                break
    else:
        i = 0

    lines_to_be_parsed = lines[i + 1 :]

    for s in strings_to_be_deleted:
        signature = signature.replace(s, " ")

    func_lines = [signature]

    indent = " " * 8

    doc = get_docstring(lines_to_be_parsed, indent=indent)

    if doc is None:
        doc = indent + "pass"

    func_lines.append(doc)

    return "\n".join(func_lines)


def create_fake_mod(dimension):
    resource = resources.files("fluidfft_builder.templates")
    with resources.as_file((resource / f"template{dimension}d.pyx")) as path:
        with open(path, encoding="utf8") as file:
            lines_text = file.read().splitlines()

    # find start class
    for i, line in enumerate(lines_text):
        if "cdef class " in line:
            lines_class = lines_text[i + 1 :]
            break

    docstring_class = get_docstring(lines_class)

    lines_function_class = lines_class[1:]

    # find functions
    functions_codes = []
    for i, line in enumerate(lines_function_class):
        if line.startswith("    def ") or line.startswith("    cpdef "):
            if " __dealloc__(" in line or "__cinit__" in line:
                continue
            functions_codes.append(get_function_code(lines_function_class[i:]))

    code = (
        f"class FFT{dimension}dFakeForDoc(object):\n"
        + docstring_class
        + "\n\n"
        + "\n\n".join(functions_codes)
        + "\n"
    )

    path_out = here / (
        f"../../../src/fluidfft/fft{dimension}d/"
        f"fake_mod_fft{dimension}d_for_doc.py"
    )

    if path_out.exists():
        old_text = path_out.read_text()
        if old_text == code:
            return

    path_out.write_text(code, encoding="utf8")


def create_fake_modules():
    for dimension in "23":
        create_fake_mod(dimension)
