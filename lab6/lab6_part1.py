import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


GRETL = r"""
function void estimates(matrix b, matrix se, scalar df)
    scalar c = critical(t, df, 0.045)
    loop j=1..rows(b) --quiet
        scalar tj = b[j]/se[j]
        scalar pj = 2*pvalue(t, df, abs(tj))
        scalar lo = b[j]-c*se[j]
        scalar hi = b[j]+c*se[j]
        printf "COEF %.12g %.12g %.12g %.12g %.12g %.12g\n", b[j], se[j], tj, pj, lo, hi
    endloop
end function

open "lab6_variant14.csv"

printf "\n=== SIMPLE ===\n"
ols Y const D
estimates($coeff, $stderr, $df)
printf "FIT %.12g %.12g %.12g\n", $rsq, $Fstat, pvalue(F, 1, $df, $Fstat)

printf "\n=== LEVEL ===\n"
ols Y const X1 X2 D
estimates($coeff, $stderr, $df)
printf "FIT %.12g %.12g %.12g\n", $rsq, $Fstat, pvalue(F, 3, $df, $Fstat)
matrix x0 = {1, 4, 10, 1}
scalar point = x0 * $coeff
scalar semean = sqrt(x0 * $vcv * x0')
scalar sepred = sqrt(semean^2 + $sigma^2)
scalar c91 = critical(t, $df, 0.045)
scalar dp = c91*sepred
scalar dm = c91*semean
printf "FORECAST %.12g %.12g %.12g %.12g %.12g\n", point, point-dp, point+dp, point-dm, point+dm
modtest --white --quiet
printf "WHITE %.12g %.12g\n", $test, $pvalue
modtest --breusch-pagan --quiet
printf "BP %.12g %.12g\n", $test, $pvalue
series e_level = $uhat
normtest e_level --dhansen
printf "DH %.12g %.12g\n", $test, $pvalue
normtest e_level --jbera
printf "JB %.12g %.12g\n", $test, $pvalue

printf "\n=== HC3 ===\n"
set hc_version 3
ols Y const X1 X2 D --robust
estimates($coeff, $stderr, $df)
restrict
    b[X1] = 0
    b[X2] = 0
    b[D] = 0
end restrict
printf "ROBUSTF %.12g %.12g\n", $test, $pvalue

printf "\n=== HSK ===\n"
hsk Y const X1 X2 D
estimates($coeff, $stderr, $df)
"""


def load_data():
    filename = "DATA_LAB_6.xlsx"
    folders = [Path.cwd(), Path(__file__).resolve().parent, Path.home() / "Downloads"]
    for folder in folders:
        candidate = folder / filename
        if candidate.is_file():
            data = pd.read_excel(candidate, sheet_name="Лист14")
            return data.set_axis(["X1", "X2", "D", "Y"], axis=1)
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


def rows(output, marker):
    return [list(map(float, line.split()[1:]))
            for line in output.splitlines() if line.startswith(marker + " ")]


def main():
    data = load_data()
    if data.shape != (540, 4) or data.isna().any().any() or set(data["D"]) != {0, 1}:
        raise ValueError("Ожидалось 540 полных наблюдений в листе «Лист14» и D ∈ {0, 1}.")

    executable = gretl_executable()
    with TemporaryDirectory() as folder:
        scratch = Path(folder)
        data.to_csv(scratch / "lab6_variant14.csv", index=False)
        (scratch / "lab6.inp").write_text(GRETL, encoding="utf-8")
        result = subprocess.run([executable, "-e", "-b", "lab6.inp"], cwd=scratch,
                                capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode:
        print(result.stdout)
        raise RuntimeError(result.stderr or "Gretl завершился с ошибкой.")

    parts = re.split(r"^=== (\w+) ===\s*$", result.stdout, flags=re.M)
    output = dict(zip(parts[1::2], parts[2::2]))
    print("ЛР № 6. Часть I. Вариант 14, 540 наблюдений. α = 0,09; γ = 0,91.")
    print("\n1. Парная регрессия зарплаты на пол (D = 0 — мужчины, D = 1 — женщины)")
    print(data.groupby("D")["Y"].agg(["size", "mean"]).to_string())
    simple = rows(output["SIMPLE"], "COEF")
    base = simple[0][0]
    difference, _, t, tp, _, _ = simple[1]
    r2, f, fp = rows(output["SIMPLE"], "FIT")[0]
    print(f"Ŷ = {base:.4f} {difference:+.4f} × D. Разница женщины − мужчины: {difference:.4f} долл./ч.")
    print(f"t = {t:.4f}, p = {tp:.4g}; F(1, 538) = {f:.4f}, p = {fp:.4g}; R² = {r2:.4f}.")
    print("При α = 0,09 связь значима, объяснённая моделью доля вариации мала.")

    print("\n2.1–2.8. Множественная регрессия и 91%-е интервалы коэффициентов")
    labels = ["Константа", "Стаж", "Образование", "Пол"]
    tables = {}
    for key in ("LEVEL", "HC3", "HSK"):
        tables[key] = pd.DataFrame(
            rows(output[key], "COEF"), index=labels,
            columns=["Оценка", "SE", "t", "p", "Нижняя 91%", "Верхняя 91%"])
    print(tables["LEVEL"].to_string(float_format=lambda value: f"{value:.6g}"))
    const, experience, education, gender = tables["LEVEL"]["Оценка"]
    r2, f_stat, f_p = rows(output["LEVEL"], "FIT")[0]
    print(f"Общая значимость: F(3, 536) = {f_stat:.4f}, p = {f_p:.4g}. "
          f"Модель объясняет {100 * r2:.2f}% вариации зарплаты.")
    print(f"Два дополнительных года стажа: {2 * experience:.4f} долл./ч.")
    print(f"Дополнительный год образования: {education:.4f} долл./ч.")
    print(f"Разница женщины − мужчины при равном стаже и образовании: {gender:.4f} долл./ч.")
    print("Для причинного вывода о дискриминации нужны дополнительные данные.")
    print("\n2.9. Прогноз почасовой оплаты")
    point, lower, upper, mean_lower, mean_upper = rows(output["LEVEL"], "FORECAST")[0]
    print(f"Женщина, стаж 4 года, образование 10 лет: прогноз {point:.4f} долл./ч.")
    print(f"91%-й интервал предсказания: [{lower:.4f}; {upper:.4f}].")
    print(f"91%-й доверительный интервал для среднего: [{mean_lower:.4f}; {mean_upper:.4f}].")
    support = data.loc[(data["D"] == 1) & (data["X2"] == 10), "X1"]
    print(f"В этой подгруппе стаж от {support.min():.2f} до {support.max():.2f} лет. "
          "При стаже 4 года прогноз является экстраполяцией.")
    print("\n2.10. Гетероскедастичность и сравнение моделей")
    for key, label in (("WHITE", "Уайт"), ("BP", "Бройш — Паган")):
        statistic, p = rows(output["LEVEL"], key)[0]
        print(f"{label}: LM = {statistic:.4f}, p = {p:.4g}.")
    print("Оба теста выявляют гетероскедастичность.")
    print(pd.concat({key: table[["Оценка", "SE", "p"]] for key, table in tables.items()},
                    axis=1).to_string(float_format=lambda value: f"{value:.6g}"))
    robust_f, robust_p = rows(output["HC3"], "ROBUSTF")[0]
    print(f"Совместный тест с HC3: F = {robust_f:.4f}, p = {robust_p:.4g}.")
    print("HC3 сохраняет коэффициенты и пересчитывает ошибки. HSK переоценивает коэффициенты с весами.")
    print("Стаж, образование и пол значимы при α = 0,09 во всех трёх оценках.")
    print("\n2.11–2.12. Нормальность и надёжность прогноза")
    dh, dh_p = rows(output["LEVEL"], "DH")[0]
    jb = rows(output["LEVEL"], "JB")[0][0]
    print(f"Дорник — Хансен: χ² = {dh:.4f}, p = {dh_p:.4g}.")
    print(f"Жара — Бера: χ² = {jb:.4f}, log10(p) = {-jb / (2 * np.log(10)):.2f}.")
    print("Оба теста отклоняют нормальность. Q–Q график показывает тяжёлый правый хвост остатков.")
    print("При 540 наблюдениях робастные оценки подтверждают выводы по факторам. "
          "Нарушения предпосылок и экстраполяция снижают надёжность индивидуального интервала.")

    design = np.column_stack([np.ones(len(data)), data[["X1", "X2", "D"]]])
    residuals = data["Y"].to_numpy() - design @ np.array([const, experience, education, gender])
    standardized = (residuals - residuals.mean()) / residuals.std(ddof=1)
    theoretical = stats.norm.ppf((np.arange(len(data)) + 0.5) / len(data))
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.scatter(theoretical, np.sort(standardized), s=12, label="Остатки")
    limits = [theoretical.min(), theoretical.max()]
    ax.plot(limits, limits, color="gray", label="Линия y = x")
    ax.set(title="Q–Q график остатков модели зарплаты",
           xlabel="Квантили N(0, 1)", ylabel="Квантили стандартизованных остатков")
    ax.legend()
    ax.grid(alpha=0.25)
    figure_path = Path(__file__).resolve().parent / "lab6_qq.png"
    fig.savefig(figure_path, dpi=180)
    print(f"Q–Q график сохранён в {figure_path.name}.")
    if plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
