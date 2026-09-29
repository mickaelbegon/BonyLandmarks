"""Point d'entrée absolu pour PyInstaller.

Lancer ``src/bonylandmarks/main.py`` comme script casse les imports relatifs
(``from .client import ...``) : « attempted relative import with no known
parent package ». On passe donc par le package installé/collecté.
"""

from bonylandmarks.main import main

if __name__ == "__main__":
    main()
