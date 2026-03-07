Getting started
===============

This is where you describe how to get set up on a clean install, including the
commands necessary to get the raw data (using the `sync_data_from_s3` command,
for example), and then how to make the cleaned, final data sets.

Recommended setup on Windows::

    python -m pip install -r requirements.txt
    python -m dvc repro

If PowerShell reports that ``dvc`` is not recognized, the package was likely
installed into the user site and its console script was placed under
``%APPDATA%\Python\Python313\Scripts``. Add that directory to ``PATH`` to use
``dvc`` directly, or keep using ``python -m dvc ...`` to bypass the PATH
requirement.
