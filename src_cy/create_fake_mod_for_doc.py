

import os

here = os.path.abspath(os.path.split(__file__)[0])


def get_doc(lines, indent='    '):
    doc = None

    # find start doc
    for i, line in enumerate(lines):
        # search for end of code
        if len(line) - len(line.lstrip(' ')) < len(indent):
            break

        if line.startswith(indent + '"""'):
            lines_to_be_parsed = lines[i+1:]
            doc = line
            break

    if doc is None or len(doc.split('"""')) > 2:
        return doc

    # find end doc
    for i, line in enumerate(lines_to_be_parsed):
        doc += '\n' + line
        if '"""' in line:
            break

    return doc

strings_to_be_deleted = [
    ' int ', ' view3df_t ', ' view3dc_t ',
    ' view2df_t ', ' view2dc_t ']


def get_function_code(lines):

    signature = lines[0].replace(' cpdef ', ' def ')

    if not signature.endswith('):'):
        for i, line in enumerate(lines[1:]):
            signature += ' ' + line.strip()
            if line.endswith('):'):
                break
    else:
        i = 0

    lines_to_be_parsed = lines[i+1:]

    for s in strings_to_be_deleted:
        signature = signature.replace(s, ' ')

    func_lines = [signature]

    indent = ' ' * 8

    doc = get_doc(lines_to_be_parsed, indent=indent)

    if doc is None:
        doc = indent + 'pass'

    func_lines.append(doc)

    return '\n'.join(func_lines)


def create_fake_mod(dimension):

    with open(os.path.join(
            here, 'template{dim}d_mako.pyx'.format(dim=dimension)), 'r') as f:
        lines_text = f.read().splitlines()

    # find start class
    for i, line in enumerate(lines_text):
        if 'cdef class ' in line:
            lines_class = lines_text[i+1:]
            break

    # get docstring of the class
    doc_class = get_doc(lines_class)

    lines_function_class = lines_class[1:]

    # find functions
    functions_codes = []
    for i, line in enumerate(lines_function_class):
        if line.startswith('    def ') or line.startswith('    cpdef '):
            if ' __dealloc__(' in line or '__cinit__' in line:
                continue
            functions_codes.append(get_function_code(lines_function_class[i:]))

    code = (
        'class FFT{dim}dFakeForDoc(object):\n'.format(dim=dimension) +
        doc_class + '\n\n' + '\n\n'.join(functions_codes) + '\n')

    name = '../fluidfft/fft{dim}d/fake_mod_fft{dim}d_for_doc.py'.format(
        dim=dimension)

    with open(os.path.join(here, name), 'w') as f:
        f.write(code)

for dim in range(2, 4):
    create_fake_mod(dim)
