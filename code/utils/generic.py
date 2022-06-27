def isnotebook():
    """Returns true if the method was called in a notebook."""
    try:
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except NameError:
        return False