import subprocess
import sys
import platform
import re
import os

PYTHON_REQUIRED = "3.11"


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
    # 1. Tentative via nvidia-smi
    try:
        result = subprocess.run(
            'cmd /c "nvidia-smi"',
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            match = re.search(r"CUDA Version:\s+(\d+\.\d+)", result.stdout)
            if match:
                print("🧠 CUDA détecté via nvidia-smi")
                return match.group(1)
            else:
                print("⚠️ nvidia-smi exécuté mais version CUDA non trouvée")
        else:
            print(f"⚠️ nvidia-smi a échoué (code {result.returncode})")
    except Exception as e:
        print(f"❌ Erreur nvidia-smi : {e}")

    # 2. Tentative via nvcc
    try:
        result = subprocess.run(
            "nvcc --version",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            match = re.search(r"release (\d+\.\d+)", result.stdout)
            if match:
                print("🧠 CUDA détecté via nvcc")
                return match.group(1)
            else:
                print("⚠️ nvcc exécuté mais version CUDA non trouvée")
        else:
            print(f"⚠️ nvcc a échoué (code {result.returncode})")
    except Exception as e:
        print(f"❌ Erreur nvcc : {e}")

    print("❌ Aucun environnement CUDA détecté")
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


def map_cuda_to_cu_tag(cuda):
    try:
        major, minor = map(int, cuda.split("."))
        if major == 12 and minor >= 8:
            return "cu128"
        elif major == 12 and minor >= 6:
            return "cu126"
        elif major == 11 and minor == 8:
            return "cu118"
    except Exception:
        pass
    return None


def install_pytorch():
    print("🔥 Installation de PyTorch...")

    cuda = detect_cuda_version()
    cu_tag = map_cuda_to_cu_tag(cuda) if cuda else None

    if cu_tag:
        print(f"✅ GPU NVIDIA détecté (CUDA {cuda}) → {cu_tag}")
        url = f"https://download.pytorch.org/whl/{cu_tag}"
        run(f"pip install torch torchvision torchaudio --index-url {url}")
        return

    if detect_amd_rocm():
        print("✅ GPU AMD ROCm 6.3 détecté")
        run("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3")
        return

    print("💻 Aucun GPU compatible détecté. Installation CPU.")
    run("pip install torch torchvision torchaudio")


def main():
    print("🚀 Installation du projet Rakuten...")

    check_python_version()

    print("📦 Mise à jour de pip...")
    run(f"{sys.executable} -m pip install --upgrade pip setuptools wheel")

    install_pytorch()

    print("📚 Installation des dépendances...")
    run("pip install -r requirements.txt")

    print("🔧 Configuration du kernel Jupyter...")
    run("pip install ipykernel")
    run("python -m ipykernel install --user --name=rakuten-3.10 --display-name='Rakuten (Python 3.10)'")

    print("🧪 Test PyTorch...")
    subprocess.run([
        "python", "-c",
        "import torch; "
        "print(f'✅ torch {torch.__version__} installé'); "
        "print(f'🎮 CUDA dispo: {torch.cuda.is_available()}'); "
        "print(f'🔧 GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '💻 Mode CPU')"
    ], check=True)

    print("\n🎉 Installation terminée !")
    print("👉 Lance Jupyter avec : jupyter lab")


if __name__ == "__main__":
    main()
