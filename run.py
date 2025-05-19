"""
Local testing script for the dashboard app.
This is useful for debugging and testing changes before deployment.
"""
import os
from dashboard_app import app

if __name__ == '__main__':
    # Get port from environment variable or use default (8050)
    port = int(os.environ.get('PORT', 8050))
    
    # Run the app
    app.run_server(
        debug=True,
        host='0.0.0.0',
        port=port
    ) 