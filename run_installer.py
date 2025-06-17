import platform
import subprocess
import os

def main():
    os_type = platform.system()

    if os_type == "Windows":
        script = "install/install.bat"
        shell = True  # Nécessaire sous Windows pour exécuter les .bat
    elif os_type in ["Linux", "Darwin"]:
        script = "./install/install.sh"
        shell = False
    else:
        print(f"Système non reconnu : {os_type}")
        return

    if not os.path.exists(script):
        print(f"Le script '{script}' est introuvable.")
        return

    try:
        subprocess.run(script, shell=shell, check=True)
        print(f"{script} exécuté avec succès.")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de {script} : {e}")

if __name__ == "__main__":
    main()
