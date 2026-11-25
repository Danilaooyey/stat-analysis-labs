import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GRETL = r"""
function void intervals(matrix b, matrix se, scalar df)
    scalar c = critical(t, df, 0.025)
    loop j=1..rows(b) --quiet
        scalar tj = b[j]/se[j]
        scalar pj = 2*pvalue(t, df, abs(tj))
        printf "COEF %.12g %.12g %.12g %.12g %.12g %.12g\n", b[j], se[j], pj, b[j]-c*se[j], b[j]+c*se[j], 2*c*se[j]
    endloop
end function

open "LAB_RAB_5_DATA.xlsx" --sheet=1
setobs 1 1 --time-series

printf "\n=== M1 ===\n"
ols Y const X1
printf "FIT %.12g %.12g %.12g\n", $rsq, 1-(1-$rsq)*($nobs-1)/$df, $dw
modtest --breusch-pagan --quiet
printf "BP %.12g %.12g\n", $test, $pvalue
series e1 = $uhat
normtest e1 --dhansen
printf "DH %.12g %.12g\n", $test, $pvalue
reset --quiet
printf "RESET_STATS %.12g %.12g\n", $test, $pvalue
intervals($coeff, $stderr, $df)

printf "\n=== M2 ===\n"
ols Y const X2
printf "FIT %.12g %.12g %.12g\n", $rsq, 1-(1-$rsq)*($nobs-1)/$df, $dw
modtest --breusch-pagan --quiet
printf "BP %.12g %.12g\n", $test, $pvalue
series e2 = $uhat
normtest e2 --dhansen
printf "DH %.12g %.12g\n", $test, $pvalue
reset --quiet
printf "RESET_STATS %.12g %.12g\n", $test, $pvalue
intervals($coeff, $stderr, $df)

printf "\n=== M3 ===\n"
ols Y const X1 X2
printf "FIT %.12g %.12g %.12g\n", $rsq, 1-(1-$rsq)*($nobs-1)/$df, $dw
modtest --breusch-pagan --quiet
printf "BP %.12g %.12g\n", $test, $pvalue
series e3 = $uhat
normtest e3 --dhansen
printf "DH %.12g %.12g\n", $test, $pvalue
reset --quiet
printf "RESET_STATS %.12g %.12g\n", $test, $pvalue
intervals($coeff, $stderr, $df)

printf "\n=== FORECAST51 ===\n"
smpl 1 49
ols Y const X1 X2
fcast 50 50 --static
smpl full

printf "\n=== FORECAST52 ===\n"
dataset addobs 1
X1[51] = 52
X2[51] = 5 + 3*sin(51)
smpl 1 50
ols Y const X1 X2
fcast 51 51 --static

printf "\n=== SIM_OLS ===\n"
open "LR5_heteroskedasticity.xlsx"
ols Y const X
modtest --breusch-pagan --quiet
printf "BP %.12g %.12g\n", $test, $pvalue
intervals($coeff, $stderr, $df)

printf "\n=== SIM_HC3 ===\n"
set hc_version 3
ols Y const X --robust
intervals($coeff, $stderr, $df)
"""


def find_workbook():
    filename = "LAB_RAB_5_DATA.xlsx"
    folders = [Path.cwd(), Path(__file__).resolve().parent, Path.home() / "Downloads"]
    for folder in folders:
        candidate = folder / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Поместите {filename} рядом со скриптом.")


def gretl_executable():
    executable = os.getenv("GRETLCLI") or shutil.which("gretlcli")
    if executable:
        return executable
    for folder in ("ProgramFiles", "ProgramFiles(x86)"):
        if os.getenv(folder):
            candidate = Path(os.environ[folder]) / "gretl" / "gretlcli.exe"
            if candidate.is_file():
                return str(candidate)
    raise FileNotFoundError("Установите Gretl или укажите путь к gretlcli в GRETLCLI.")


def numeric_rows(section, label):
    return [list(map(float, line.split()[1:]))
            for line in section.splitlines() if line.startswith(label + " ")]


def coefficient_table(section, factors):
    columns = ["Оценка", "SE", "p", "Нижняя 95%", "Верхняя 95%", "Ширина ДИ"]
    return pd.DataFrame(numeric_rows(section, "COEF"), index=["const", *factors], columns=columns)


def print_table(table):
    print(table.to_string(float_format=lambda value: f"{value:.6g}"))


def main():
    workbook = find_workbook()
    executable = gretl_executable()
    data = pd.read_excel(workbook, sheet_name="ВАРИАНТ 13")
    if data.shape != (50, 3) or data.isna().any().any():
        raise ValueError("Ожидалось 50 полных наблюдений в листе «ВАРИАНТ 13».")
    x = np.arange(1, 51)
    error = np.random.default_rng(6311).normal(0, 0.35 * x)
    simulation = pd.DataFrame({"X": x, "Y": 10 + 2 * x + error})
    output = Path(__file__).resolve().parent
    saved_data = output / "LR5_heteroskedasticity.xlsx"
    simulation.to_excel(saved_data, index=False, engine="xlsxwriter")

    with TemporaryDirectory() as folder:
        scratch = Path(folder)
        shutil.copyfile(workbook, scratch / workbook.name)
        shutil.copyfile(saved_data, scratch / saved_data.name)
        (scratch / "lab5.inp").write_text(GRETL, encoding="utf-8")
        result = subprocess.run([executable, "-e", "-b", "lab5.inp"], cwd=scratch,
                                capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode:
        print(result.stdout)
        raise RuntimeError(result.stderr or "Gretl завершился с ошибкой.")

    parts = re.split(r"^=== (\w+) ===\s*$", result.stdout, flags=re.M)
    sections = dict(zip(parts[1::2], parts[2::2]))
    models = [("M1", ["X1"]), ("M2", ["X2"]), ("M3", ["X1", "X2"])]
    print("ЛР № 5. Часть I. Лист «ВАРИАНТ 13», 50 наблюдений.")
    print("\nЗадания 2–4 и 6.1. Коэффициенты трёх моделей и 95%-е интервалы")
    print_table(pd.concat({name: coefficient_table(sections[name], factors)
                           for name, factors in models}, names=["Модель", "Параметр"]))
    fit_rows, test_rows = [], []
    for name, _ in models:
        section = sections[name]
        r2, adjusted_r2, dw = numeric_rows(section, "FIT")[0]
        fit_rows.append((name, r2, adjusted_r2, dw))
        for label, test in (("BP", "Бреуш — Паган"), ("DH", "Дорник — Хансен"), ("RESET_STATS", "RESET")):
            statistic, p = numeric_rows(section, label)[0]
            test_rows.append((name, test, statistic, p))
    print_table(pd.DataFrame(fit_rows, columns=["Модель", "R²", "Скорр. R²", "DW"]).set_index("Модель"))
    print("\nЗадание 5. Диагностика моделей")
    print_table(pd.DataFrame(test_rows, columns=["Модель", "Тест", "Статистика", "p"]).set_index("Модель"))
    print("M3 имеет наибольший скорректированный R².")
    print("Тесты Бреуша — Пагана, Дорника — Хансена и RESET в M1–M3 "
          "не отклоняют соответствующие H₀ при α = 0,05.")
    print("DW ниже 1,5 у M1 и M2. У M3 DW входит в диапазон [1,5; 2,5].")
    print("По порядку X1 у M2 виден выраженный тренд остатков. У M3 тренд не прослеживается.")
    print("В M3 интервалы наклонов уже, чем у тех же факторов в одиночных моделях.")
    print("\nЗадание 6.2. Прогнозы по модели M3")
    for label, index in (("FORECAST51", 50), ("FORECAST52", 51)):
        match = re.search(rf"^ +{index} +(.+?)\s+-\s+(\d+\.\d+)\s*$", sections[label], re.M)
        values = list(map(float, match[1].split()))
        print(f"X1 = {index + 1}: Ŷ = {values[-3]:.2f}; 95%-й интервал [{values[-1]:.2f}; {float(match[2]):.2f}].")
    print("При X1 = 51 последняя строка исключена из обучения. При X1 = 52 "
          "использовано продолжение X2 = 5 + 3 sin(X1 − 1).")
    print("95%-й интервал предсказания относится к одному новому значению Y.")
    print("\nЗадания 7.1–7.3. Синтетическая выборка и робастные ошибки HC3")
    print_table(pd.concat({label: coefficient_table(sections[key], ["X"])
                           for label, key in (("МНК", "SIM_OLS"), ("МНК HC3", "SIM_HC3"))},
                          names=["Метод", "Параметр"]))
    lm, p = numeric_rows(sections["SIM_OLS"], "BP")[0]
    print(f"Бреуш — Паган: LM = {lm:.6f}, p = {p:.4g}.")
    print("В симуляции тест Бреуша — Пагана выявляет гетероскедастичность. "
          "HC3 сохраняет оценки, сужает интервал константы и расширяет интервал наклона.")
    print("Интервалы HC3 оценивают коэффициенты синтетической модели. "
          "Интервал в пункте 6.2 оценивает разброс отдельного будущего Y.")
    print(f"Данные симуляции сохранены в {saved_data.name}.")

    fig, axes = plt.subplots(2, 3, figsize=(10, 5), layout="constrained")
    for i, (name, factors) in enumerate(models):
        regressors = np.column_stack([np.ones(len(data)), data[factors]])
        beta = np.linalg.lstsq(regressors, data["Y"], rcond=None)[0]
        fitted = regressors @ beta
        residuals = data["Y"] - fitted
        for ax, values, label in ((axes[0, i], fitted, "Предсказанные значения"),
                                  (axes[1, i], data["X1"], "X1 (порядок строк)")):
            ax.scatter(values, residuals, s=16)
            ax.axhline(0, color="gray")
            ax.set(title=name, xlabel=label, ylabel="Остаток")
            ax.grid(alpha=0.25)
    fig.savefig(output / "lab5_residuals.png", dpi=180)
    print("График остатков сохранён в lab5_residuals.png.")
    if plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
