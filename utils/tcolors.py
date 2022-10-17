class terminalColor:
    """Attributes generates colored sting
    """
    info = '\033[93m' + "[INFO] " + '\033[0m'
    warn = '\033[38;5;214m' + "[WARN] " + '\033[0m'
    err = '\033[91m' + "[ERROR] " + '\033[0m'
    success = '\033[92m' + "[OK] " + '\033[0m'
    help = '\033[94m' + "[HELP] " + '\033[0m'
    train = '\u001b[42;1m' + "[TRAIN]" + '\u001b[0m '
    debug = '\u001b[43;1m' + '[DEBUG]' + '\u001b[0m '
    thread = '\u001b[35m' + '[THREAD]' + '\u001b[0m '
    process = '\u001b[34;1m' + '[PROCESS]' + '\u001b[0m '