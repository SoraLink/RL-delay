def _init_openmpi():
    """Pre-load libmpi.dll and register OpenMPI distribution."""
    import os
    import ctypes
    if os.name != 'nt' or 'OPENMPI_HOME' in os.environ:
        return
    try:
        openmpi_home = os.path.abspath(os.path.dirname(__file__))
        openmpi_bin = os.path.join(openmpi_home, 'bin')
        os.environ['OPENMPI_HOME'] = openmpi_home
        os.environ['PATH'] = ';'.join((openmpi_bin, os.environ['PATH']))
        ctypes.cdll.LoadLibrary(os.path.join(openmpi_bin, 'libmpi.dll'))
    except Exception:
        pass

_init_openmpi()