#!C:\Users\stezer\Desktop\projectForTwitter\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'problog==2.1.0.39','console_scripts','problog'
__requires__ = 'problog==2.1.0.39'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('problog==2.1.0.39', 'console_scripts', 'problog')()
    )
