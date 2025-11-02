# Jupyter configuration for PyCharm/WSL connectivity

c = get_config()

# Bind to all network interfaces so Windows can connect to WSL
c.ServerApp.ip = '0.0.0.0'

# Allow connections from Windows
c.ServerApp.allow_remote_access = True

# Don't open browser automatically (PyCharm will connect directly)
c.ServerApp.open_browser = False

# Set the notebook directory
c.ServerApp.notebook_dir = '/mnt/c/Users/maria/Desktop/fruit-classifier-AI-project'
