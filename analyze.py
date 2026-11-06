from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'data'/'JobSatisfaction.csv'
OUT=ROOT/'outputs'; FIG=ROOT/'figures'
OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
df=pd.read_csv(SOURCE)
expected=['SatisfactionLevel','LastEvaluation','NumberProject','AverageMonthlyHours','TimeSpendCompany','WorkAccident','LeftCompany','PromotionLast5years','Department','Salary']
assert set(expected)==set(df.columns),f'Unexpected columns: {set(df.columns)^set(expected)}'
assert len(df)>0
before=len(df)
# No rows removed unless duplicates are exact; retain duplicate records because they can represent distinct employees.
for c in ['Department','Salary']:
    df[c]=df[c].astype('string').str.strip().str.lower()
assert df[expected].notna().all().all(),'Missing values found: inspect before encoding.'
for c in ['WorkAccident','LeftCompany','PromotionLast5years']:
    assert set(df[c].unique()).issubset({0,1}),f'{c} has unexpected values'
# Frequency (quantile) bins; use ranks to ensure 3 balanced bins even when tied values occur.
# Rank tie-break by input order; boundaries can split tied values, documented as a tradeoff.
labels=['Low','Avg','High']
processed=df.copy()
quantile_info=[]
for c in ['SatisfactionLevel','LastEvaluation','AverageMonthlyHours']:
    ranks=df[c].rank(method='first')
    processed[c+'_Bin']=pd.qcut(ranks,q=3,labels=labels).astype('string')
    for lab in labels:
        vals=df.loc[processed[c+'_Bin']==lab,c]
        quantile_info.append({'Feature':c,'Bin':lab,'Count':len(vals),'Min':vals.min(),'Max':vals.max()})
# One categorical item per original feature, including discretized continuous features.
# Satisfaction bin is the target item, but no model or rules are computed in Weeks 1-3.
items=pd.DataFrame(index=df.index)
for c in ['SatisfactionLevel','LastEvaluation','AverageMonthlyHours']:
    items[c]=processed[c+'_Bin'].map(lambda v:f'{c}_{v}')
for c in ['NumberProject','TimeSpendCompany']:
    items[c]=df[c].map(lambda v:f'{c}_{v}')
for c in ['WorkAccident','LeftCompany','PromotionLast5years']:
    items[c]=df[c].map(lambda v:f'{c}_{v}')
for c in ['Department','Salary']:
    items[c]=df[c].map(lambda v:f'{c}_{v}')
boolean=pd.get_dummies(items, dtype=int)
assert boolean.isin([0,1]).all().all()
assert (boolean.sum(axis=1)==len(expected)).all()
processed.to_csv(OUT/'cleaned_binned_dataset.csv',index=False)
boolean.to_csv(OUT/'boolean_feature_matrix.csv',index=False)
pd.DataFrame(quantile_info).to_csv(OUT/'bin_summary.csv',index=False)
summary=pd.DataFrame({'Feature':df.columns,'Type':[str(df[c].dtype) for c in df.columns], 'Missing':[int(df[c].isna().sum()) for c in df.columns], 'Unique':[int(df[c].nunique()) for c in df.columns], 'Values_or_range':[', '.join(map(str,sorted(df[c].unique()))) if df[c].nunique()<=15 else f'{df[c].min()} to {df[c].max()}' for c in df.columns]})
summary.to_csv(OUT/'data_audit.csv',index=False)
df.describe(include='all').transpose().to_csv(OUT/'descriptive_statistics.csv')
for c in ['Salary','PromotionLast5years','WorkAccident','Department','LeftCompany']:
    counts=df[c].astype(str).value_counts().sort_index()
    fig,ax=plt.subplots(figsize=(8,4.5));bars=ax.bar(counts.index,counts.values)
    ax.set_title(f'{c} distribution');ax.set_ylabel('Employee count');ax.tick_params(axis='x',rotation=35 if c=='Department' else 0)
    for bar in bars: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),str(int(bar.get_height())),ha='center',va='bottom',fontsize=8)
    fig.tight_layout();fig.savefig(FIG/f'{c}_distribution.png',dpi=150);plt.close(fig)
print('Rows',before,'Columns',len(df.columns),'Boolean columns',boolean.shape[1]);print('Salary',df.Salary.value_counts().to_dict());print('Left',df.LeftCompany.value_counts().to_dict());print('Bins',pd.DataFrame(quantile_info).to_string(index=False))
