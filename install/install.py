# install.py

import subprocess
import sys
import platform
import re
import os

PYTHON_REQUIRED = "3.10"

def check_python_version():
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if not version.startswith(PYTHON_REQUIRED):
        print(f"❌ Erreur: Python {PYTHON_REQUIRED}.x requis, version détectée: {version}")
        sys.exit(1)
    print(f"✅ Python {version} détecté")

def run(cmd):
    print(f"🔧 {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def detect_cuda_version():
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, universal_newlines=True)
        match = re.search(r"CUDA Version:\s+(\d+\.\d+)", output)
        return match.group(1) if match else None
    except Exception:
        return None

def detect_amd_rocm():
    if platform.system() != "Linux":
        return False
    try:
        if os.path.exists("/dev/kfd"):
            output = subprocess.check_output("rocminfo", stderr=subprocess.DEVNULL, universal_newlines=True)
            return "gfx" in output.lower()
    except Exception:
        pass
    return False

def install_pytorch():
    print("🔥 Installation de PyTorch...")

    cuda_versions = {
        "12.1": "cu121",
        "12.0": "cu120",
        "11.8": "cu118"
    }

    cuda = detect_cuda_version()
    if cuda and cuda in cuda_versions:
        cu_tag = cuda_versions[cuda]
        url = f"https://download.pytorch.org/whl/{cu_tag}"
        print(f"✅ GPU NVIDIA détecté - CUDA {cuda}")
        run(f"pip install torch torchvision torchaudio --index-url {url}")
        return

    if detect_amd_rocm():
        print("✅ GPU AMD ROCm 6.3 détecté")
        run("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3")
        return

    print("💻 Aucun GPU compatible détecté, installation CPU.")
    run("pip install torch torchvision torchaudio")

def main():
    print("🚀 Installation du projet Rakuten...")

    check_python_version()

    print("📦 Mise à jour de pip...")
    run("pip install --upgrade pip setuptools wheel")

    install_pytorch()

    print("📚 Installation des dépendances...")
    run("pip install -r requirements.txt")

    print("🔧 Configuration du kernel Jupyter...")
    run("pip install ipykernel")
    run("python -m ipykernel install --user --name=rakuten-3.10 --display-name='Rakuten (Python 3.10)'")

    print("🧪 Test de l'installation PyTorch...")
    subprocess.run([
        "python", "-c",
        "import torch; "
        "print(f'✅ PyTorch {torch.__version__} installé'); "
        "print(f'🎮 CUDA disponible: {torch.cuda.is_available()}'); "
        "print(f'🔧 GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '💻 Mode CPU activé')"
    ], check=True)

    print("\n🎉 Installation terminée ! Le projet est prêt.")
    print("👉 Pour démarrer : jupyter lab")

if __name__ == "__main__":
    main()
