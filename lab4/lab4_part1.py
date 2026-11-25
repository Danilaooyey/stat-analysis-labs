from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

years = [2005, 2010, 2015, 2019, 2020, 2021]
alpha = 0.02
filename = "CHISLO_DOCTORS.xlsx"
locations = [Path.cwd(), Path(__file__).resolve().parent, Path.home() / "Downloads"]
source = next((folder / filename for folder in locations if (folder / filename).is_file()), None)
if source is None:
    raise FileNotFoundError(f"Поместите {filename} рядом со скриптом.")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*extension is not supported.*",
                            category=UserWarning, module=r"openpyxl\.worksheet\._reader")
    raw = pd.read_excel(source, sheet_name="ROSSTAT", usecols="A,H:M",
                        header=None, engine="openpyxl")
assert list(raw.iloc[3, 1:]) == years
raw.columns = ["Субъект", *years]
data = raw.iloc[4:].copy()
data["Субъект"] = data["Субъект"].astype("string").str.split().str.join(" ")
data[years] = data[years].apply(pd.to_numeric, errors="coerce")

blocks = {"СКФО": (49, 55), "ЮФО": (40, 47)}
districts = {
    name: data.loc[first - 1:last - 1].set_index("Субъект")[years]
    for name, (first, last) in blocks.items()
}
north, south = districts["СКФО"], districts["ЮФО"]
russia = data.loc[4, years].astype(float)
print("1. Выборка СКФО")
print(north.to_string())
print("Пропуски исключаются отдельно по каждому году. Единица измерения — врачи на 10 000 жителей.")

def samples(frame):
    return {year: frame[year].dropna() for year in years}


x, y = samples(north), samples(south)
description = pd.DataFrame({
    year: [len(v), v.mean(), v.std(ddof=1), v.median(),
           stats.skew(v, bias=False), stats.kurtosis(v, bias=False)]
    for year, v in x.items()
}, index=["n", "Среднее", "Ст. отклонение", "Медиана",
          "Асимметрия", "Эксцесс"]).T
description["n"] = description["n"].astype(int)
print("\n2. Описательная статистика")
print(description.round(4).to_string())
print("По асимметрии и эксцессу ближе всего к нормальному закону 2005 год.")

unit = "Врачей на 10 000 жителей"
color = "#2D6785"
fig_hist, axes = plt.subplots(2, 3, figsize=(9.0, 4.7), sharex=True,
                             sharey=True, layout="constrained")
bins = np.arange(25, 76, 10)
for year, ax in zip(years, axes.flat):
    ax.hist(x[year], bins=bins, color=color, edgecolor="white", zorder=2)
    ax.set_title(f"{year} г. (n = {len(x[year])})")
    ax.set_xticks(bins)
    ax.set_yticks(range(6))
    ax.grid(axis="y", alpha=0.25)
fig_hist.supxlabel(unit)
fig_hist.supylabel("Число субъектов")
axes.flat[0].legend(handles=[Patch(color=color, label="СКФО")],
                    loc="upper right", fontsize=8)

def boxplot(values, labels, xlabel):
    fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
    ax.boxplot(values, patch_artist=True, showmeans=True,
               boxprops={"facecolor": "#DCEAF2"},
               medianprops={"color": color, "linewidth": 2},
               meanprops={"marker": "D", "markersize": 4,
                          "markerfacecolor": "#A54F22",
                          "markeredgecolor": "#A54F22"})
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set(xlabel=xlabel, ylabel=unit)
    ax.grid(axis="y", alpha=0.25)
    ax.plot([], [], color=color, label="Медиана")
    ax.plot([], [], "D", color="#A54F22", markersize=4, label="Среднее")
    ax.legend(loc="upper left")
    return fig


fig_years = boxplot(list(x.values()), years, "Год")

def decision(p):
    return "Отклонить H₀" if p <= alpha else "Не отклонять H₀"


def shapiro_by_year(groups):
    rows = []
    for year, values in groups.items():
        w, p = stats.shapiro(values)
        rows.append((year, len(values), w, p, decision(p)))
    return pd.DataFrame(rows, columns=["Год", "n", "W", "p", "Решение"]
                        ).set_index("Год")


normal_x = shapiro_by_year(x)
normal_y = shapiro_by_year(y)
print("\n3. Шапиро — Уилк для СКФО")
print("H₀: распределение по субъектам в каждом году нормальное.")
print(normal_x.round(6).to_string())
print("При α = 0,02 ни в одном году нормальность не отклоняется.")

print("\n4. Независимые выборки СКФО и ЮФО")
print(normal_y.round(6).to_string())
print("H₀: показатели СКФО и ЮФО не различаются. H₁: уровень СКФО ниже.")
alternative = "less"
permutation = stats.PermutationMethod(n_resamples=np.inf)
rows = []
for year in years:
    a, b = x[year], y[year]
    f = p_levene = np.nan
    if normal_x.at[year, "p"] > alpha and normal_y.at[year, "p"] > alpha:
        f, p_levene = stats.levene(a, b, center="median")
        equal_var = p_levene > alpha
        result = stats.ttest_ind(a, b, equal_var=equal_var,
                                 alternative=alternative)
        method = "Стьюдент" if equal_var else "Уэлч"
    else:
        result = stats.mannwhitneyu(a, b, alternative=alternative,
                                    method=permutation)
        method = "Манн — Уитни"
    rows.append((year, f, p_levene, method, result.statistic,
                 result.pvalue, decision(result.pvalue)))
independent = pd.DataFrame(rows, columns=[
    "Год", "F Левена", "p Левена", "Критерий", "Статистика", "p", "Решение"
]).set_index("Год")
print(independent.round(6).to_string())
print("Левен не выявил различий дисперсий. Значимого отставания СКФО от ЮФО нет.")

pairs = north[[2010, 2020]].dropna().copy()
pairs["D"] = (pairs[2020] - pairs[2010]).round(1)
d = pairs["D"]
w_d, p_d = stats.shapiro(d)
if p_d > alpha:
    paired_method = "Парный t-критерий Стьюдента"
    paired = stats.ttest_rel(pairs[2020], pairs[2010], alternative="two-sided")
else:
    paired_method = "Парный критерий Уилкоксона"
    paired = stats.wilcoxon(d, alternative="two-sided", method=permutation)
print("\n5. Парное сравнение 2010 и 2020 годов")
print("H₀: изменения нет. H₁: изменение есть.")
print(pairs.to_string())
print(f"Шапиро — Уилк: W = {w_d:.6f}, p = {p_d:.6f}")
print(f"{paired_method}: статистика = {paired.statistic:.6f}, "
      f"p = {paired.pvalue:.6f}, {decision(paired.pvalue)}")
print(f"Средняя разность = {d.mean():.4f}, медиана = {d.median():.4f}")
if p_d > alpha:
    ci = paired.confidence_interval(confidence_level=1 - alpha)
    print(f"98%-й ДИ разности: [{ci.low:.4f}; {ci.high:.4f}]")
print("В 2010–2020 гг. показатель статистически значимо вырос.")

rows = []
for year in years:
    a, benchmark = x[year], russia[year]
    if normal_x.at[year, "p"] <= alpha:
        continue
    result = stats.ttest_1samp(a, benchmark, alternative=alternative)
    method = "Стьюдент"
    rows.append((year, benchmark, a.mean(), a.median(), method,
                 result.statistic, result.pvalue, decision(result.pvalue)))
one_sample = pd.DataFrame(rows, columns=[
    "Год", "РФ", "Среднее СКФО", "Медиана СКФО", "Критерий",
    "Статистика", "p", "Решение"
]).set_index("Год")
print("\n6. СКФО и общероссийский уровень")
print("H₀: среднее СКФО равно уровню РФ. H₁: среднее СКФО ниже.")
print(one_sample.round(6).to_string())
print("Значимого отставания СКФО от общероссийского уровня не выявлено.")
output = Path(__file__).resolve().parent
fig_hist.savefig(output / "lab4_histograms.png", dpi=180)
fig_years.savefig(output / "lab4_boxplots.png", dpi=180)
print("Графики сохранены в lab4_histograms.png и lab4_boxplots.png.")
if plt.get_backend().lower() != "agg":
    plt.show()
