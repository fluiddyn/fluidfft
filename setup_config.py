"""Do not modify this file to modify your configuration. Instead, copy
site.cfg.default to site.cfg and modify this file.

"""
import os
import sys

try:  # python 3
    from configparser import ConfigParser
except ImportError:  # python 2.7
    from ConfigParser import ConfigParser

sections_libs = ["fftw3", "fftw3_mpi", "cufft", "pfft", "p3dfft"]
sections = sections_libs + ["environ"]


def get_default_config():
    """Generate default configuration."""

    config = ConfigParser()

    use = {k: "False" for k in sections_libs}
    # by default we built nothing and we just use pyfftw!
    # use['fftw3'] = 'True'

    for section in sections_libs:
        config.add_section(section)
        config.set(section, "use", use[section])
        config.set(section, "dir", "")
        config.set(section, "include_dir", "")
        config.set(section, "library_dir", "")

    config.add_section("environ")

    return config


def make_site_cfg_default_file():
    """Write the default configuration to site.cfg.default."""

    config = get_default_config()

    with open("site.cfg.default", "w") as configfile:
        config.write(configfile)


def get_config():
    """Check for site-specific configuration file saved as either:

    1. site.cfg in source directory, or
    2. $HOME/.fluidfft-site.cfg

    and read if found, else revert to default configuration.

    Returns
    -------
    dict

    """
    config = get_default_config()

    user_dir = "~user" if sys.platform == "win32" else "~"
    configfile_user = os.path.expanduser(
        os.path.join(user_dir, ".fluidfft-site.cfg")
    )

    for configfile in ("site.cfg", configfile_user):
        if os.path.exists(configfile):
            print("Parsing", configfile)
            config.read(configfile)
            break
    else:
        print("Using default configuration.")
        print(
            "Copy site.cfg.default -> site.cfg or $HOME/.fluidfft-site.cfg "
            "to specify site specific libraries."
        )

    config_dict = {}
    for section in config.sections():
        if section not in sections:
            raise ValueError("Unexpected library in site.cfg: {}".format(section))

        section_dict = {}
        for option in config.options(section):
            value = config.get(section, option)
            if section == "environ":
                option = option.upper()
            else:
                if option == "use":
                    value = value.lower()
                    if not (section == "fftw3" and value in ("mkl", "mkl_rt")):
                        value = config.getboolean(section, option)
                else:
                    value = os.path.expanduser(value)
                    value = os.path.expandvars(value)

            section_dict[option] = value

        config_dict[section] = section_dict
        if "use" in section_dict and section_dict["use"]:
            print(section + ": ")
            for k, v in section_dict.items():
                k = "{}: ".format(k).rjust(25)
                print(k, v)

    return config_dict


def parse_config():
    """Parse configuration dictionary for special cases."""
    config = get_config()

    TMP = os.getenv("FFTW3_INC_DIR")
    if TMP is not None:
        print("Use value in FFTW3_INC_DIR")
        config["fftw3"]["include_dir"] = TMP

    TMP = os.getenv("FFTW3_LIB_DIR")
    if TMP is not None:
        print("Use value in FFTW3_LIB_DIR")
        config["fftw3"]["library_dir"] = TMP

    lib_flags_dict = {}

    # See https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor
    if config["fftw3"]["use"] == "mkl":
        lib_flags_dict["fftw3"] = [
            "mkl_intel_ilp64",
            "mkl_sequential",
            "mkl_core",
        ]
    elif config["fftw3"]["use"] == "mkl_rt":
        lib_flags_dict["fftw3"] = ["mkl_rt", "pthread", "m", "dl"]

    lib_dirs_dict = {}
    for lib in sections_libs:
        cfg = config[lib]
        if len(cfg["dir"]) > 0:
            lib_dirs_dict[lib] = os.path.join(cfg["dir"], "lib")

        path = cfg["library_dir"]
        if len(path) > 0:
            lib_dirs_dict[lib] = path

    return config, lib_flags_dict, lib_dirs_dict


if __name__ == "__main__":
    make_site_cfg_default_file()
