import pandas as pd

df = pd.read_csv("./data/raw/results.txt")
print(df.shape)
print(df.columns.tolist())
print(df.isna().mean().sort_values(ascending=False).head(30))

cols = [
    "ConvertedCompYearly",
    "Country",
    "WorkExp",
    "YearsCode",
    "EdLevel",
    "Age",
    "DevType",
    "Employment",
    "OrgSize",
    "RemoteWork",
    "Industry",
    "ICorPM",
    "LanguageHaveWorkedWith",
]
print(df[cols].isna().mean().sort_values(ascending=False))
print(df[cols].dtypes)
print(df["ConvertedCompYearly"].notna().sum(), "people reported a salary")

raw = df.copy()

duplicates = raw.drop(columns="ResponseId").duplicated().sum()
print("duplicates : ", duplicates)

df = raw.drop_duplicates(subset=raw.columns.drop("ResponseId")).copy()
print(df.shape)
dup_rows = raw[raw.drop(columns="ResponseId").duplicated(keep=False)]
print(
    dup_rows["ConvertedCompYearly"].notna().sum(), "of the duplicate rows have a salary"
)
print(dup_rows.isna().mean(axis=1).mean().round(2), "average share of empty answers")

# removing the rows where compsalary is empty its critical for final prediction
before = len(df)
df = df[df["ConvertedCompYearly"].notna()]
print("removed:", before - len(df), "| remaining:", len(df))

print(df["MainBranch"].value_counts())
# keeping only the professional developer for better accuracy
before = len(df)
df = df[df["MainBranch"] == "I am a developer by profession"]
print("remove :", before - len(df), " | remaining:", len(df))

# dropping unemployed/retired responses - a reported salary from someone without
# a current job isn't representative of what we're trying to predict (current
# compensation based on current role, remote work, org size, etc.)
# NOTE: "Student" is kept here - the EDA compares junior salaries for
# Employed vs Student and filters Student out itself for later analyses
before = len(df)
df = df[~df["Employment"].isin(["Not employed", "Retired"])]
print("remove :", before - len(df), " | remaining:", len(df))

# picking up only the required columns
df = df[cols].copy()
print(df.shape)

# checking for outliers
print(df["ConvertedCompYearly"].describe())
print(df["ConvertedCompYearly"].quantile([0.01, 0.05, 0.95, 0.99]))

# identifying outliers country based
low_earners = df[df["ConvertedCompYearly"] < 5000]
print(len(low_earners), " People earn under $5000")
print(low_earners["Country"].value_counts().head(15))
print(low_earners["ConvertedCompYearly"].describe())

# we need to check and drop outliers countrywise because 2000 salary in nepal is not an outlier but in usa it is
country_median = df.groupby("Country")["ConvertedCompYearly"].transform("median")

# flaging salaries below 10% of the median w.r.t country
too_low = df["ConvertedCompYearly"] < 0.10 * country_median
print(too_low.sum(), "salaries are below 10% of their country's median")
print(df[too_low]["Country"].value_counts().head(10))

# droping lows and highs
high = df["ConvertedCompYearly"].quantile(0.99)
df = df[~too_low & (df["ConvertedCompYearly"] <= high)]


print(f"upper cut-off: ${high:,.0f}")
print("removed:", before - len(df), "| remaining:", len(df))
print(df["ConvertedCompYearly"].describe())

# checking for the wrong median values becuase of less respondants
print(df.nsmallest(15, "ConvertedCompYearly")[["Country", "ConvertedCompYearly"]])
print((df["ConvertedCompYearly"] < 1000).sum(), "people still under $1,000")

# putting in floor value to drop obvious mistakes eg 1$
before = len(df)
df = df[df["ConvertedCompYearly"] >= 1000].copy()
print("removed:", before - len(df), "| remaining:", len(df))
print(df["ConvertedCompYearly"].min())

# removing countries with less entries
counts = df["Country"].value_counts()
print("countries before: ", counts)
rare = counts[counts < 50].index
print(len(rare), "countries have fewer than 50 people")

# grouping in others category
df["Country"] = df["Country"].replace(rare, "Other")
print("countries after:", df["Country"].nunique())
print((df["Country"] == "Other").sum(), "people now in 'Other'")

print(df.isna().sum())

before = len(df)
# droping WorkExp YearsCode EdLevel where missing
df = df.dropna(subset=["WorkExp", "YearsCode", "EdLevel"])
print("removed: ", before - len(df), " | remaining: ", len(df))

# capping impossible experience values (some respondents entered more than 50 years)
df[["WorkExp", "YearsCode"]] = df[["WorkExp", "YearsCode"]].clip(upper=50)

# standardising the different "didn't answer" labels into a single Unknown label
df = df.replace(
    {
        "I don’t know": "Unknown",  # curly ’ as in the survey
        "Prefer not to say": "Unknown",
        "I prefer not to say": "Unknown",
    }
)

# filling the rest where missing too many to drop
fill_cols = ["OrgSize", "RemoteWork", "ICorPM", "Industry", "LanguageHaveWorkedWith"]
df[fill_cols] = df[fill_cols].fillna("Unknown")

print(df.isna().sum().sum())


print(df.shape)
print(df["ConvertedCompYearly"].describe())

df.to_csv("data/processed/clean.csv", index=False)
print("saved", df.shape)
