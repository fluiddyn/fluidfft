"""Do not modify this file to modify your configuration. Copy
site.cfg.example to site.cfg and modify this file.

"""

import os

try:  # python 3
    from configparser import ConfigParser
except:  # python 2.7
    from ConfigParser import ConfigParser


def get_default_config():

    config = ConfigParser()

    sections = ['fftw', 'fftw-mpi', 'cufft', 'pfft', 'p3dfft']

    use = {k: 'False' for k in sections}
    use['fftw'] = 'True'

    for section in sections:
        config.add_section(section)
        config.set(section, 'use', use[section])
        config.set(section, 'dir', '')
        config.set(section, 'include_dir', '')
        config.set(section, 'library_dir', '')

    return config


def make_site_cfg_example_file():

    config = get_default_config()

    with open('site.cfg.example', 'wb') as configfile:
        config.write(configfile)


def get_config():
    config = get_default_config()

    if os.path.exists('site.cfg'):
        config.read('site.cfg')

    config_dict = {}
    for section in config.sections():
        section_dict = {}
        for option in config.options(section):
            value = config.get(section, option)
            if option == 'use':
                value = value.lower()
                if value not in ['true', 'false']:
                    raise ValueError('"use" should be "True" of "False".')
                value = value == 'true'
            section_dict[option] = value
        config_dict[section] = section_dict

    return config_dict


if __name__ == '__main__':
    make_site_cfg_example_file()
