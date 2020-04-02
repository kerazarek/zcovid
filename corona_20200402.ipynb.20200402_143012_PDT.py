#!/usr/bin/env python
# coding: utf-8

# # covid stuff
# 
# ----------------------------------------
# 
# - **created** by z: `2020-03-30`
# - last **updated**: `2020-04-02T13:39:50PDT`

# ## _preamble_
# 
# #### import packages

# In[1]:


import pathlib
import requests
import re
import math
import numpy as np
import pandas as pd
from IPython.display import display as disp
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.tools as tls
from plotly.offline import iplot


# #### disable request warning

# In[2]:


from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


# #### directories

# In[13]:


# cwd = pathlib.Path.cwd()
# cwd
# docs_dir = pathlib.Path("/private/var/mobile/Library/Mobile Documents/iCloud~AsheKube~Carnets/Documents")
# data_dir = docs_dir / "data"

# params
on_ipad = False # false if on mac
read_from_csse = True # read from the CSSE dir, not zcovid data dir
overwrite_csvs = False # (only if not CSSE) overwrite CSVs if already existing

# basic paths
HOME = pathlib.Path.home()
zcovid_root = HOME / "Dropbox/code/github/zcovid"
csse_data_dir = zcovid_root / "COVID-19/csse_covid_19_data"
csse_tsdata_dir = csse_data_dir / "csse_covid_19_time_series"
icloud_root = pathlib.Path("/private/var/mobile/Library/Mobile Documents")
carnets_dir = icloud_root / "iCloud~AsheKube~Carnets/Documents"

# where data will be saved/loaded
# data_dir = pathlib.Path("/Users/zarek/Dropbox/code/github/zcovid/data")
# data_dir = pathlib.Path("/Users/zarek/Dropbox/code/github/zcovid/COVID-19/csse_covid_19_data/csse_covid_19_time_series")

if on_ipad:
    data_dir = carnets_dir / "data" 
elif read_from_csse:
    data_dir = csse_tsdata_dir
    overwrite_csvs = False
else:
    data_dir = zcovid_root / "data"
    
if not data_dir.is_dir():
    data_dir.mkdir()
    print(">>> created dir {}".format(data_dir))


# #### URLs

# In[14]:


# base URL for data downloads
base_tsdata_url = "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data"


# #### regexes

# In[15]:


date_re = re.compile(r"\d+/\d+/\d+")


# #### time-series keys

# In[16]:


# for file names and saving data
tskeys = [
    'confirmed_US',
    'confirmed_global',
    'deaths_US',
    'deaths_global',
    'recovered_global'
]


# #### utility functions

# In[17]:


# sinum(): format number SI

SINUM_PREFIXES = {
    -6: {'short': "a", 'long': "atto"},
    -5: {'short': "f", 'long': "femto"},
    -4: {'short': "p", 'long': "pico"},
    -3: {'short': "n", 'long': "nano"},
    -2: {'short': "μ", 'long': "micro"},
    -1: {'short': "m", 'long': "milli"},
    0: {'short': " ", 'long': "-"},
    1: {'short': "k", 'long': "kilo"},
    2: {'short': "M", 'long': "mega"},
    3: {'short': "G", 'long': "giga"},
    4: {'short': "T", 'long': "tera"},
    5: {'short': "P", 'long': "peta"},
    6: {'short': "E", 'long': "exa"}
}

def sinum(num, unit='B', fmt="{coef:.3f} {pfx}{unit}", long_pfx=False, strip_zeros=True, binary=False, verbose=False):

    # check inputs
    assert isinstance(num, (int, float))
    assert isinstance(unit, str)
    
    # verbosity...
    if verbose:
        print(f">>> sinum(num={num!r}, unit={unit!r}, fmt={fmt!r}, long_pfx={long_pfx}, strip_zeros={strip_zeros}, binary={binary}, verbose={verbose})")
    def _verb(name, value):
        if verbose:
            print("\t{:<12s} = {:>20s}".format(
                name, 
                value if isinstance(value, str) else repr(value)
            ))
            
    # binary mods
    if binary:
        log_base = 1024
        unit = f"i{unit!s}"        
    else:
        log_base = 1000
        unit = str(unit)
        
    # order of magnitude
    if num == 0:
        oom = 0
    else:
        oom = math.floor(math.log(-num if num < 0 else num, log_base))
    if oom < -6:
        oom = -6
    if oom > 6:
        oom = 6
    _verb('oom', oom)
    
    # coefficient
    coef = num / (log_base ** oom)
    _verb('coef', coef)
    
    # SI prefix
    pfx = SINUM_PREFIXES[oom]['long' if long_pfx else 'short']
    _verb('pfx', pfx)
    
    # string out
    out = fmt.format(coef=coef, pfx=pfx, unit=unit)
    # if strip_zeros:
    #     while re.match(r"\d+\.0{2,}", out):
    #         # out = re.sub(r"(?:\d+(?:\.0*)?)0", ' ', out)
    #         out = re.sub(r"(\d+\.0*?)(0)\b", '\1 ', out)
    _verb('out', out)

    return out


# tests
if False:
    print(sinum(82457891234, verbose=True), end='\n\n')
    print(sinum(82457891234, verbose=True, unit=''), end='\n\n')
    print(sinum(82457891234, verbose=True, unit='bloops'), end='\n\n')
    print(sinum(82457891234, binary=True, verbose=True), end='\n\n')
    print(sinum(0.00082457891234, verbose=True), end='\n\n')
    print(sinum(824578912342345.2345, verbose=True), end='\n\n')
    print(sinum(0.00082457891234, binary=True, verbose=True), end='\n\n')
    print(sinum(824578912342345.2345, binary=True, verbose=True), end='\n\n')
    print(sinum(824578912342345.2345, binary=True, verbose=True, fmt="{coef:12.5f} // {pfx} // {unit}"), end='\n\n')
    print(sinum(824578912342345.2345, binary=True, fmt="{coef:12.5f} // {pfx} // {unit}"), end='\n\n')
    print(sinum(1, verbose=True), end='\n\n')
    print(sinum(-1, verbose=True), end='\n\n')
    print(sinum(0, verbose=True), end='\n\n')
    print(sinum(-82457891234, verbose=True), end='\n\n')
    print(sinum(8786996786798967896872457891234, verbose=True), end='\n\n')


# ## load data
# 
# #### initialize data container `d`

# In[20]:


# a dict with tskeys as keys to uniform dicts
d = {}
for tsk in tskeys:
    d[tsk] = {}
disp(d)


# #### download data from github, save to file

# In[19]:


# same load and save process for each tskey
print(f">>> loading CSVs, saving in ``{data_dir}''\n")
for tsk in tskeys:
    
    print(">>> getting data for '{}'".format(tsk))
    
    # download URL from github
    d[tsk]['url'] = f"{base_tsdata_url}/csse_covid_19_time_series/time_series_covid19_{tsk}.csv"
    
    # CSV to load from and/or write to
    d[tsk]['csv'] = data_dir / f"time_series_covid19_{tsk}.csv"
    
    # check for existing file
    if d[tsk]['csv'].is_file() and d[tsk]['csv'].stat().st_size > 0:
        if overwrite_csvs:
            print(f"--> CSV for '{tsk}' already exists and will be overwritten")
        else:
            print(f"--> CSV for '{tsk}' already exists and won't be overwritten, skipping")    
            continue # skip to next tskey
            
    # web request
    d[tsk]['req'] = requests.get(d[tsk]['url'], auth=('user', 'pass'))
    
    # CSV contents as a big string
    d[tsk]['raw'] = d[tsk]['req'].content.decode() # decode cause it opens as bytes
    
    # save the CSV
    with d[tsk]['csv'].open('w') as f:
        print("--> writing ``.../{}'' ... ".format(d[tsk]['csv'].name), end='')
        f.write(d[tsk]['raw'])
        print("wrote {}\n".format(sinum(d[tsk]['csv'].stat().st_size)))
        
    # ditch that big string
    del d[tsk]['raw']
    
# disp(d)


# #### load data from CSV just saved

# In[ ]:


# iterate tskeys, loading from CSV saved above
for tsk in tskeys:
    d[tsk]['df'] = pd.read_csv(d[tsk]['csv'])
    print(f"--> read CSV data for {tsk}")
    
# create backup of d
d_BAK = d.copy()


# ## clean up data
# 
# #### add index columns `d[tsk]['df']` (actual dataframe for the key), then reorder as desired

# In[ ]:


# column name substitutions
col_subs = {
    'Province_State': 'subregion',
    'Province/State': 'subregion',
    'Country_Region': 'region',
    'Country/Region': 'region',
    'Long_': 'long'    
} 

# columns to move to the beginning (in order)
priority_cols = [
    'locid',
    'region',
    'subregion',
    'combined_key',
    'lat',
    'long',
    'population'
]

# collect all columns since different dataframes dont have same columns
all_indx_cols = []
all_date_cols = []

# iterate through tskeys, cleaning up each
for tsk in tskeys:
    
    print(f">>> cleaning up dataframe for '{tsk}'")
    
    # add other index cols
    d[tsk]['df']['tskey'] = tsk
    d[tsk]['df']['domain'] = tsk.split('_')[1]
    d[tsk]['df']['datum'] = tsk.split('_')[0]
    d[tsk]['df']['locid'] = d[tsk]['df'].index
    
    d[tsk]['all_cols'] = list(d[tsk]['df'].columns)
    
    # clean up column names
    for i, c in enumerate(d[tsk]['all_cols']):
        if c in col_subs:
            c = col_subs[c]
        c = c.lower()
        d[tsk]['all_cols'][i] = c
    # print(d[tsk]['all_cols'])
    
    # get column subsets
    d[tsk]['df'].columns = d[tsk]['all_cols']
    d[tsk]['date_cols'] = list(filter(date_re.match, d[tsk]['all_cols']))
    d[tsk]['indx_cols'] = [i for i in d[tsk]['all_cols'] if i not in d[tsk]['date_cols']]

    # reorder columns
    col_idxs = list(range(len(d[tsk]['indx_cols'])))
    for col in priority_cols[::-1]:
        if col in d[tsk]['indx_cols']:
            idx = d[tsk]['indx_cols'].index(col)
            col_idxs.remove(idx)
            col_idxs.insert(0, idx)
    print(col_idxs)
    d[tsk]['indx_cols'] = [d[tsk]['indx_cols'][i] for i in col_idxs]
        
    # add to all_indx_cols
    for col in d[tsk]['indx_cols']:
        if col not in all_indx_cols:
            all_indx_cols.append(col)

    # add to all_date_cols
    for col in d[tsk]['date_cols']:
        if col not in all_date_cols:
            all_date_cols.append(col)

    # save dataframe with reordered columns
    d[tsk]['all_cols'] = [*d[tsk]['indx_cols'], *d[tsk]['date_cols']]
    d[tsk]['df'] = d[tsk]['df'][d[tsk]['all_cols']]

# d[tsk]

print(all_indx_cols)
print(all_date_cols)

disp(d[tsk]['df'])


# #### create backups of `d[tsk]['df']`

# In[ ]:


for tsk in tskeys:
    # print(d[tsk]['indx_cols'])
    # if type(d[tsk]['df'].columns).__name__ == 'Index':
    if ('df_BAK' not in d[tsk]) or isinstance(d[tsk]['df'].columns, pd.Index):
        d[tsk]['df_BAK'] = d[tsk]['df'].copy()


# #### convert `d[tsk]['df']` such that rows are dates and columns are multi-index

# In[ ]:


for tsk in tskeys:
    
    # create multiindex dataframe (df with just index cols)
    mindx_df = d[tsk]['df_BAK'][d[tsk]['indx_cols']]
    # create multiindex
    mindx = pd.MultiIndex.from_frame(mindx_df)
    # create new dataframe, old df transposed
    d[tsk]['df'] = d[tsk]['df_BAK'][d[tsk]['date_cols']].transpose()
    # add the new multiindex
    d[tsk]['df'].columns = mindx
    # convert index from str to datetime
    d[tsk]['df'].index = pd.to_datetime(d[tsk]['df'].index)
    
# mindx_df
# mindx

# disp(d[tskeys[0]]['df'])


# #### add levels to multi-index of `d[tsk]['df']` so they are uniform across all

# In[ ]:


for tsk in tskeys:  # [tskeys[1]]
    
    # # re-collect multiindex from dataframe
    # mindx = d[tsk]['df'].columns
    # # re-create mindx df
    # mindx_df = mindx.to_frame()
    # # reindex to reorder columns and add empty keys for any missing
    # mindx_df = mindx_df.reindex(columns=all_indx_cols)                
    # # # below is alternative way to do reindex
    # # for col in all_indx_cols:
    # #     if col not in mindx_df.columns:
    # #         mindx_df = mindx_df.assign(**{col: np.nan})
    # # change that dataframe back into a multiindex
    # mindx = pd.MultiIndex.from_frame(mindx_df)
    # # assign the multiindex back into the original data
    # d[tsk]['df'].columns = mindx

    # below does all the above as a one-liner (expanded for readability)
    d[tsk]['df'].columns = pd.MultiIndex.from_frame(
        d[tsk]['df'].columns
            .to_frame()
            .reindex(columns=all_indx_cols)
    )

# print(all_indx_cols)
# print(mindx_df.columns)
# mindx_df

# disp(d[tskeys[3]]['df'])


# ## compile data
# 
# #### create joined dataframe from individual `d[tsk]['df']`s

# In[ ]:


# initialize new key in d, with first df as its df
d['all'] = {'df': d[tskeys[0]]['df']}

# iterate through rest of dfs, joining into 'all' df
for tsk in tskeys[1:]:
    d['all']['df'] = d['all']['df'].join(d[tsk]['df'])
    
# print(d['all']['df'].shape)
disp(d['all']['df'])


# ## compute secondary data
# 
# #### columns to be computed

# In[22]:


# calcs = {
#     'deaths_per_confirmed': lambda df:
# }


# #### actually compute them

# In[24]:



print(df.shape)
sdf = df.iloc[:, (glv('region') == "United Kingdom") & (glv('domain') == "global")]
print(sdf.shape)

# sdf
# sdf.columns.to_frame()
disp(sdf)
disp(df)


# ## plot stuff!
# 
# #### ...

# In[ ]:


df = d['all']['df']
glv = df.columns.get_level_values

# df
print(list(glv('region').unique().sort_values()))
# df.iloc[:, glv('region') == "US"]
# df.iloc[:, (glv('region') == "US") | (glv('region') == "United Kingdom")]
# df.iloc[:, glv('region') in ["US", "United Kingdom"]]
# df.columns
# df.columns.to_frame().loc[:, 'region'].unique()
# glv(1)
# glv('region') == "US"
# usdf = df.iloc[:, (glv('region') == "US") & np.isnan(glv('subregion'))]
# usdf
df.iloc[:, (glv('region') == "US") & (glv('subregion').isna())]
# list(usdf.columns.to_frame().subregion.unique())


# In[ ]:


# all data
df = d['all']['df']
glv = df.columns.get_level_values

# plotting params
prms = {
    'domain': ["global"],
    'datum': ["confirmed"],
    'region': ["US", "United Kingdom", "China"],
    'subregion': [np.nan]
}

# subset data
# sdf = df.iloc[:, (glv('domain') == 'global') & (glv('datum') == 'confirmed')].iloc[:, :3]
sbool = np.ones(df.shape[1], bool)
# print(sum(sbool))
for prm, vals in prms.items():
    sbool = sbool & (glv(prm).isin(vals))
    # print(sbool.shape)
sdf = df.iloc[:, sbool]
sglv = sdf.columns.get_level_values
psdf = sdf.copy()
psdf.columns = list(sglv('region'))

disp(sdf)


# In[ ]:


# plot it
fignum = 0
fig = plt.figure(num=fignum, figsize=(15, 10), dpi=80)

# plt.plot(sdf)
# plt.legend(sglv('region'))
print(psdf.columns)
plt.plot(psdf)
plt.legend(list(psdf.columns))
plt.show()


# In[ ]:


import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Scatter(
        x=psdf.index,
        y=psdf.US
    ),
])
fig.show()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'qt')
fig, axs = plt.subplots(2,1, figsize = (10,6))
axs[0].plot(psdf)
# fig
# ax.plot(psdf.US)
plotly_fig = tls.mpl_to_plotly(fig) ## convert 
iplot(plotly_fig)


# In[ ]:


df.iloc[:, glv('region') == "US"] #.columns.to_frame().subregion.unique()


# In[ ]:


fignum = 0
fig = plt.figure(num=fignum, figsize=(30, 50), dpi=80)

df = d['all']['df']
glv = df.columns.get_level_values
# df = df.iloc[:, (glv('region') == "US") & (glv('subregion').isna() | (glv('subregion') == "California"))]
df = df.iloc[:, (glv('region') == "US") & (glv('subregion').isna() & (glv('subregion') == "California"))]
glv = df.columns.get_level_values

# print(list(glv('combined_key').unique()))
# print([x for x in (glv('combined_key').isna())])
# print(list(glv('combined_key')[(~glv('combined_key').isna()) & (glv('combined_key').to_series().str.contains("California"))]))
# print(list(glv('combined_key')[(glv('combined_key').isna()) | (glv('combined_key').to_series().str.contains("California"))]))
# plot_regions = plot_data.columns[[0, 1, 2]].get_level_values('region')
# plt.plot(df)
# plt.legend(plot_regions)

df

# plt.plot(df)
# plt.legend(glv('combined_key'))
# plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




