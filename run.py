import argparse
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
import venv
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
GRETL_URL = "https://downloads.sourceforge.net/project/gretl/gretl/2026d/gretl-2026d-win64.zip"
GRETL_SHA256 = "bb4cee33a5aea915815d51608c24f94c2c78053373fc9295a16468911f5171fa"
LABS = ("4", "5", "6")


def prepare_python():
    virtualenv = ROOT / ".venv"
    python = virtualenv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        print("Создаю .venv", flush=True)
        venv.EnvBuilder(with_pip=True).create(virtualenv)

    requirements = [ROOT / "requirements.txt"]
    requirements.extend(ROOT / f"lab{lab}" / "requirements.txt" for lab in LABS)
    fingerprint = hashlib.sha256(b"\0".join(path.read_bytes() for path in requirements)).hexdigest()
    stamp = virtualenv / "lab-requirements.sha256"
    if not stamp.is_file() or stamp.read_text(encoding="ascii") != fingerprint:
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements[0])],
                       cwd=ROOT, check=True)
        stamp.write_text(fingerprint, encoding="ascii")
    return python


def prepare_gretl():
    configured = os.environ.get("GRETLCLI") or shutil.which("gretlcli")
    if configured:
        return configured
    candidates = []
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        if os.environ.get(variable):
            candidates.append(Path(os.environ[variable]) / "gretl" / "gretlcli.exe")
    candidates.append(Path("/Applications/Gretl.app/Contents/MacOS/gretlcli"))
    portable = ROOT / ".tools" / "gretl" / "gretlcli.exe"
    candidates.append(portable)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    if os.name != "nt" or platform.machine().lower() not in ("amd64", "x86_64"):
        raise RuntimeError("Установите Gretl и добавьте gretlcli в PATH или задайте GRETLCLI. См. README.md.")

    print("Скачиваю Gretl 2026d для Windows x64 в .tools", flush=True)
    portable.parent.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=portable.parent.parent) as temp:
        archive = Path(temp) / "gretl.zip"
        request = Request(GRETL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=60) as response, archive.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        with archive.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != GRETL_SHA256:
                raise RuntimeError("Контрольная сумма архива Gretl не совпала. Установите Gretl вручную.")
        stage = Path(temp) / "unpacked"
        with ZipFile(archive) as package:
            package.extractall(stage)
        if not (stage / "gretl" / "gretlcli.exe").is_file():
            raise RuntimeError("В архиве отсутствует gretlcli.exe.")
        shutil.move(str(stage / "gretl"), portable.parent)
    return str(portable)


def run_lab(python, lab, environment):
    folder = ROOT / f"lab{lab}"
    log_path = folder / f"lab{lab}_output.txt"
    print(f"\nЛабораторная № {lab}", flush=True)
    command = [str(python), "-u", str(folder / f"lab{lab}_part1.py")]
    with log_path.open("w", encoding="utf-8") as log, subprocess.Popen(
        command, cwd=folder, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        if process.wait():
            raise RuntimeError(f"Ошибка при запуске ЛР № {lab}. Подробности в {log_path}.")
    print(f"Вывод сохранён в {log_path.relative_to(ROOT)}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Запуск лабораторных по статистическому анализу")
    parser.add_argument("lab", nargs="?", choices=(*LABS, "all"), default="all")
    parser.add_argument("--no-show", action="store_true", help="сохранить графики без открытия окон")
    args = parser.parse_args()
    if sys.version_info[:2] not in ((3, 12), (3, 13), (3, 14)):
        parser.exit(1, "Для закреплённых версий пакетов используйте Python 3.12, 3.13 или 3.14.\n")

    selected = LABS if args.lab == "all" else (args.lab,)
    environment = {**os.environ, "PYTHONUTF8": "1"}
    if args.no_show:
        environment["MPLBACKEND"] = "Agg"
    try:
        python = prepare_python()
        if any(lab in ("5", "6") for lab in selected):
            environment["GRETLCLI"] = prepare_gretl()
        for lab in selected:
            run_lab(python, lab, environment)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
