#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
import dvc.api
from git import Repo


def get_code_rev(project_root: Path) -> str:
    repo = Repo(project_root, search_parent_directories=True)
    return repo.head.commit.hexsha


def read_tracking_list(project_root: Path, rel_path: str = ".dvc/dvc_tracking_list.txt") -> list[str]:
    tracking_file = (project_root / rel_path)
    if not tracking_file.exists():
        raise FileNotFoundError(f"Fichier introuvable: {tracking_file}")

    items: list[str] = []
    for line in tracking_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        items.append(s)
    if not items:
        raise ValueError(f"La liste est vide: {tracking_file}")
    return items


def read_dvc_out_hash(dvc_file_rel: str, repo: str, rev: str) -> dict:
    """
    Lit un fichier .dvc et renvoie {dvc_file, tracked_path, data_rev}.
    dvc_file_rel doit être un chemin relatif à la racine du repo.
    """
    txt = dvc.api.read(dvc_file_rel, repo=repo, rev=rev)
    doc = yaml.safe_load(txt)

    outs = doc.get("outs") or []
    if not outs:
        raise ValueError(f"Aucun 'outs' dans {dvc_file_rel}")

    out0 = outs[0]
    data_rev = out0.get("md5") or out0.get("hash") or out0.get("etag")
    if not data_rev:
        raise ValueError(f"Aucun hash trouvé dans {dvc_file_rel}")

    return {
        "dvc_file": dvc_file_rel,
        "tracked_path": out0.get("path"),
        "data_rev": data_rev,
    }


def resolve_to_dvc_file(item: str, project_root: Path) -> str:
    """
    Si 'item' est déjà un .dvc, on le garde.
    Sinon, on tente de trouver un .dvc correspondant à une sortie suivie:
      - data/foo -> data/foo.dvc (convention la plus courante)
    On renvoie un chemin relatif (POSIX) depuis project_root.
    """
    p = Path(item)

    # cas: entrée = fichier .dvc
    if p.suffix == ".dvc":
        return p.as_posix()

    # cas: entrée = chemin de données, on tente <path>.dvc
    candidate = (project_root / (p.as_posix() + ".dvc"))
    if candidate.exists():
        return candidate.relative_to(project_root).as_posix()

    # sinon, on échoue clairement
    raise FileNotFoundError(
        f"Entrée '{item}' n'est pas un .dvc et aucun '{item}.dvc' trouvé sous {project_root}"
    )


def collect_data_revs_from_tracking_list(project_root: Path, repo: str, rev: str) -> dict[str, dict]:
    items = read_tracking_list(project_root)
    results: dict[str, dict] = {}

    for item in items:
        dvc_file_rel = resolve_to_dvc_file(item, project_root)
        info = read_dvc_out_hash(dvc_file_rel, repo=repo, rev=rev)
        # clé = dvc file, valeur = infos
        results[dvc_file_rel] = info

    return results


def write_json(payload: dict, output_path: Path, indent: int = 2) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, sort_keys=True)
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export lineage JSON (code_rev + DVC data hashes) from .dvc/dvc_tracking_list.txt"
    )
    p.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Racine du projet (contient .git, .dvc, etc.).",
    )
    p.add_argument(
        "--rev",
        default="HEAD",
        help="Révision Git à laquelle lire les fichiers .dvc (HEAD par défaut).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/lineage.json"),
        help="Chemin du JSON de sortie.",
    )
    p.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation JSON.",
    )
    p.add_argument(
        "--tracking-file",
        default=".dvc/dvc_tracking_list.txt",
        help="Chemin relatif du fichier de tracking (par défaut: .dvc/dvc_tracking_list.txt).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()

    # Pour dvc.api.read, repo peut être un path local
    repo_path = str(project_root)

    code_rev = get_code_rev(project_root)
    data_rev = collect_data_revs_from_tracking_list(project_root, repo=repo_path, rev=args.rev)

    payload = {
        "code_rev": code_rev,
        "git_rev_for_data": args.rev,
        "data_rev": data_rev,
        "tracking_list": args.tracking_file,
    }

    out = write_json(payload, args.out, indent=args.indent)
    print(f"✅ Lineage exporté vers: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
