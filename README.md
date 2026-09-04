# Flood-wave travel time from stream gages

A self-contained teaching notebook that pulls **every USGS gage on a river** straight from the
USGS Water Data API, downloads 15-minute **flow and stage hydrographs** for two floods, puts the
gages in upstream → downstream order with channel distances from the national hydrography
network (NLDI), and estimates the **flood-wave travel time and celerity** between gages by
matched-peak timing and lagged cross-correlation.

The worked example is **Cypress Creek (Harris County, TX)** for **Tax Day 2016** and
**Hurricane Harvey 2017**. Section 10 re-runs the entire pipeline on the **Guadalupe River
above Canyon Lake** (Hunt → Spring Branch) with a one-line configuration change, so a new
student can see exactly what to edit to point it at their own river.

No shapefiles, no GIS software, no data stored in the repo — everything is pulled live.

## Quick start

The notebook ships **already executed**, so every table and figure is visible the moment you
open it (including in GitHub's preview).

### A · Google Colab — one click, nothing to install

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jdf9/cypress-gage-travel-time/blob/main/cypress_gage_hydrographs_travel_time.ipynb)

Click the badge, then **Runtime → Run all**. The first cell installs `dataretrieval` (the only
package Colab does not already have). Full run is about 2 minutes; the data are pulled live from
`api.waterdata.usgs.gov` and `api.water.usgs.gov/nldi`.

You will see a one-time line *"No API key detected"* — ignore it; the notebook stays well inside
the anonymous rate limit. (For heavy scripted use register a free key at
<https://api.waterdata.usgs.gov/signup/> and set the `API_USGS_PAT` environment variable.)

### B · Local — macOS / Linux

```bash
git clone https://github.com/jdf9/cypress-gage-travel-time.git
cd cypress-gage-travel-time
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab            # then open cypress_gage_hydrographs_travel_time.ipynb
```

### C · Local — Windows (PowerShell)

```powershell
git clone https://github.com/jdf9/cypress-gage-travel-time.git
cd cypress-gage-travel-time
py -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter lab            # then open cypress_gage_hydrographs_travel_time.ipynb
```

> On some managed Windows machines, calling `pip.exe` or the `jupyter` shim directly is blocked
> by app control — use `python -m pip install ...` and `python -m nbconvert ...` from inside the
> activated venv instead.

## What's in the box

```
cypress_gage_travel_time/
  cypress_gage_hydrographs_travel_time.ipynb   - the notebook (ships executed, ~2 MB with figures)
  README.md                                    - this file
  requirements.txt                             - pip dependencies
  build_notebook.py                            - regenerates the .ipynb from source (see below)
```

## Notebook outline

| § | Section | What the student learns |
|---|---|---|
| 0 | Setup | `dataretrieval` ≥ 1.3, the new `waterdata` + `nldi` modules, one fixed colour palette |
| 1 | **Configuration** | The one cell to edit for a new river: HUC prefix, name pattern, tributary exclusions, event windows, max lag |
| 2 | Find the gages | `waterdata.get_monitoring_locations` with a CQL filter |
| 3 | Data availability | `get_time_series_metadata` → which gages have 15-min flow / stage during each event |
| 4 | Order + distance | NLDI upstream-main navigation, flowline set-differences → channel miles between gages |
| 5 | Download hydrographs | `get_continuous`, tidy → wide, UTC → local time, gap handling, and a **quality gate** that picks flow or stage per gage per event |
| 6 | Stacked hydrographs | The flood wave marching downstream; overlay figure |
| 7 | Peak table | Peak flow / stage, timing, crest duration, secondary peaks |
| 8 | Travel time | Matched peak-to-peak lag with a bounded search window and flags, windowed cross-correlation, celerity in m/s and mi/h, travel-time diagram |
| 9 | Interpretation | What the Cypress numbers mean: overbank storage, the Sharp Rd flat crest (Addicks spill level), why Harvey is faster than Tax Day, what to distrust |
| 10 | **Now you: Guadalupe River** | Same pipeline on a steep Hill Country river (October 2018 and July 2025 floods) + exercises |

Headline results as executed (matched-peak method):

| River | Event | Reaches matched | Distance | Travel time | Celerity |
|---|---|---|---|---|---|
| Cypress Creek | Tax Day 2016 | 5 / 5 | 35.4 mi | 35.8 h | 0.99 mi/h (0.44 m/s) |
| Cypress Creek | Harvey 2017 | 4 / 5 | 32.1 mi | 20.0 h | 1.61 mi/h (0.72 m/s) |
| Guadalupe River | October 2018 | 5 / 6 | 91.4 mi | 22.8 h | 4.02 mi/h (1.80 m/s) |
| Guadalupe River | July 2025 | 5 / 5 | 97.8 mi | 23.2 h | 4.20 mi/h (1.88 m/s) |

Rerunning will refresh the data; provisional records can change, so small differences are normal.

## Adapting it to another river

Edit the `CONFIGS` dictionary in Section 1 (or add a new entry) and set `RIVER` to its key:

```python
"guadalupe_upper": dict(
    title="Guadalupe River above Canyon Lake, TX",
    huc_prefix="12100201",          # HUC-8 (or longer) that contains the reach
    name_like="%Guadalupe Rv%",     # SQL LIKE pattern on the USGS station name
    exclude=["N Fk", "S Fk"],       # substrings that mark tributaries to drop
    tz="America/Chicago",
    events={"October 2018": ("2018-10-14", "2018-10-20"),
            "July 2025":    ("2025-07-03", "2025-07-09")},
    max_lag_hours=24,               # longest plausible reach-to-reach lag
    xcorr_window_hours=6,           # half-width of the cross-correlation template
    override_order=None, override_distance_km=None),
```

Then **Run all**. Section 10 shows the whole thing done for the Guadalupe with
`run_pipeline(CONFIGS["guadalupe_upper"])`. Tips that the notebook explains in place:

- Find the HUC and the exact station-name spelling on the USGS *Water Data for the Nation* site
  for any one gage on the river.
- Steep rivers need a shorter `max_lag_hours` / `xcorr_window_hours` than flat coastal streams.
- If NLDI navigation misorders gages (braided reaches, diversions), fill `override_order` and
  `override_distance_km` by hand.

## Re-executing the notebook headlessly

```bash
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 cypress_gage_hydrographs_travel_time.ipynb
```

`build_notebook.py` regenerates the un-executed `.ipynb` from Python source (`python build_notebook.py`),
which is the easiest way to make larger edits; run the `nbconvert` command afterwards to bake the outputs in.

## Notes on colours

Each gage keeps the same colour in every figure (colour follows the gage, never its rank), from a
fixed palette chosen to be safe for colour-vision deficiency. Every figure has a single y-axis —
flow and stage are never plotted on twin axes.
