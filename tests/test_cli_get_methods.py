import sys

from unittest.mock import patch

from fluidfft.cli_get_methods import main


def test_get_methods():

    with patch.object(sys, "argv", []):
        main()

    with patch.object(sys, "argv", ["fluidfft-get-methods", "-d", "2"]):
        main()

    with patch.object(sys, "argv", ["fluidfft-get-methods", "-d", "3"]):
        main()

    with patch.object(sys, "argv", ["fluidfft-get-methods", "-s"]):
        main()

    with patch.object(sys, "argv", ["fluidfft-get-methods", "-p"]):
        main()

    with patch.object(sys, "argv", ["fluidfft-get-methods", "-d", "3", "-s"]):
        main()
