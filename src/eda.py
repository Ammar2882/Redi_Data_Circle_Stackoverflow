import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import mannwhitneyu

df = pd.read_csv("data/processed/clean.csv")
print(df.shape)

sns.set_theme(style="whitegrid")

juniors = df[(df["WorkExp"] <= 4) & df["Employment"].isin(["Employed", "Student"])]

# 1. median salary per group
print(juniors.groupby("Employment")["ConvertedCompYearly"].agg(["count", "median"]))

# 2. statistical test: is the difference real or random?
emp = juniors.loc[juniors["Employment"] == "Employed", "ConvertedCompYearly"]
stu = juniors.loc[juniors["Employment"] == "Student", "ConvertedCompYearly"]
print("Mann-Whitney p-value:", mannwhitneyu(emp, stu).pvalue)

# 3. box plot
sns.boxplot(data=juniors, x="Employment", y="ConvertedCompYearly", showfliers=False)
plt.title("Salary of juniors (≤4 years experience): Employed vs Student")
plt.ylabel("Annual salary (USD)")
plt.savefig("outputs/figures/juniors_employed_vs_student.png", bbox_inches="tight")
plt.show()

# the test above confirms students earn significantly less than employed
# juniors, so including them would bias the compensation model toward lower
# salaries for reasons unrelated to the actual job features - drop them here
# for the rest of the analysis (unemployed/retired were already dropped in
# data_cleaning.py, since no comparison was needed to justify that)
df = df[df["Employment"] != "Student"].copy()

df["rel"] = df["ConvertedCompYearly"] / df.groupby("Country")["ConvertedCompYearly"].transform(
    "median"
)

d = df.groupby("DevType").agg(
    n=("rel", "size"), median_usd=("ConvertedCompYearly", "median"), rel=("rel", "median")
)
d = d[d["n"] >= 100].sort_values("rel")  # skip tiny groups
d["rel"].plot.barh(figsize=(9, 8))
plt.axvline(1, ls="--", color="gray")
plt.title("Salary by role, relative to own country's median")
plt.xlabel("Salary ÷ country median")
plt.tight_layout()
plt.savefig("outputs/figures/salary_by_role.png", bbox_inches="tight")
plt.show()

langs = df[df["LanguageHaveWorkedWith"] != "Unknown"].copy()
langs["lang"] = langs["LanguageHaveWorkedWith"].str.split(";")
langs = langs.explode("lang")  # one row per person per language
L = langs.groupby("lang")["rel"].agg(["median", "size"])
L = L[L["size"] >= 400].sort_values("median")  # only common languages
((L["median"] - 1) * 100).plot.barh(figsize=(9, 8))
plt.axvline(0, color="gray")
plt.title("Language pay premium vs own country's median")
plt.xlabel("% above / below country median")
plt.tight_layout()
plt.savefig("outputs/figures/language_pay_premium.png", bbox_inches="tight")
plt.show()

d = df[df["ICorPM"] != "Unknown"].copy()
d["exp_band"] = pd.cut(
    d["WorkExp"], [0, 5, 10, 15, 20, 50], labels=["1-5", "6-10", "11-15", "16-20", "21+"]
)
t = d.pivot_table(
    index="exp_band", columns="ICorPM", values="rel", aggfunc="median", observed=True
)
print(t.round(2))
t.plot(marker="o", figsize=(8, 5))
plt.title("Manager vs individual contributor at the same experience")
plt.ylabel("Salary ÷ country median")
plt.xlabel("Years of work experience")
plt.savefig("outputs/figures/manager_vs_ic.png", bbox_inches="tight")
plt.show()

print(
    "p-value:",
    mannwhitneyu(
        d.loc[d.ICorPM == "Individual contributor", "rel"],
        d.loc[d.ICorPM == "People manager", "rel"],
    ).pvalue,
)
# 1. only people who listed languages
d = df[df["LanguageHaveWorkedWith"] != "Unknown"].copy()
lists = d["LanguageHaveWorkedWith"].str.split(";")

# 2. languages used by at least 400 people
counts = lists.explode().value_counts()
langs = counts[counts >= 400].index.tolist()

# 3. one 0/1 column per language
L = pd.DataFrame({l: lists.apply(lambda x: l in x).astype(float) for l in langs}, index=d.index)

# 4. the other factors to hold equal
controls = pd.get_dummies(d[["Country", "DevType", "OrgSize"]], drop_first=True, dtype=float)
controls["WorkExp"] = d["WorkExp"]
controls["WorkExp2"] = d["WorkExp"] ** 2
controls["n_langs"] = lists.str.len()

# 5. fit the regression on log salary
X = sm.add_constant(pd.concat([controls, L], axis=1))
model = sm.OLS(np.log(d["ConvertedCompYearly"]), X).fit()

# 6. convert language effects to % with 95% intervals
ci = model.conf_int().loc[langs]
res = pd.DataFrame({"effect": model.params[langs], "low": ci[0], "high": ci[1]})
res = ((np.exp(res) - 1) * 100).sort_values("effect")

# 7. plot
fig, ax = plt.subplots(figsize=(9, 8))
sig = (res["low"] > 0) | (res["high"] < 0)
colors = ["tab:blue" if s else "lightgray" for s in sig]
ax.hlines(res.index, res["low"], res["high"], colors=colors, lw=2)
ax.scatter(res["effect"], res.index, color=colors, zorder=3)
ax.axvline(0, color="gray")
ax.set_title("Salary difference linked to each language, other factors held equal")
ax.set_xlabel("% salary difference (same country, experience, role, company size)")
plt.tight_layout()
plt.savefig("outputs/figures/language_regression.png", bbox_inches="tight")
plt.show()