
from __future__ import print_function

with open('template3d_mako.pyx', 'r') as f:
    text = f.read()

lines_text = text.splitlines()

# find start class
for i, line in enumerate(lines_text):
    if 'cdef class ' in line:
        lines_class = lines_text[i+1:]
        break


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

doc_class = get_doc(lines_class)

strings_to_be_deleted = [
    ' DTYPEf_t[:, :, ::1] ',  ' DTYPEc_t[:, :, ::1] ', ' int ']


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

lines_function_class = lines_class[1:]

# find functions
functions_codes = []
for i, line in enumerate(lines_function_class):
    if line.startswith('    def ') or line.startswith('    cpdef '):
        if ' __dealloc__(' in line or '__cinit__' in line:
            continue
        functions_codes.append(get_function_code(lines_function_class[i:]))

code = (
    '\nclass FakeFFTClassForDoc(object):\n' + doc_class + '\n' +
    '\n\n'.join(functions_codes) + '\n')

with open('fake_mod.py', 'w') as f:
    f.write(code)
