"""
download_metro_data.py
======================
Run this ONCE to download all metro datasets into data/external/.
Then run cross_metro_comparison.ipynb.

Usage:
    python download_metro_data.py

Requirements:
    pip install pandas requests pyarrow
"""

import os
import sys
import pandas as pd
import requests
from io import StringIO
from pathlib import Path

DATA_DIR = Path("data/external")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url, filepath, description, chunk_size=8192):
    """Download a file with progress indicator."""
    print(f"\n{'='*60}")
    print(f"Downloading: {description}")
    print(f"URL: {url[:100]}{'...' if len(url)>100 else ''}")
    print(f"Saving to: {filepath}")
    print(f"{'='*60}")
    
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024*1024)
        print(f"  Already exists ({size_mb:.1f} MB) — skipping")
        return True
    
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        
        downloaded = 0
        with open(filepath, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = 100 * downloaded / total
                    mb = downloaded / (1024*1024)
                    print(f"\r  {mb:.1f} MB ({pct:.0f}%)", end='', flush=True)
                else:
                    mb = downloaded / (1024*1024)
                    print(f"\r  {mb:.1f} MB", end='', flush=True)
        
        size_mb = filepath.stat().st_size / (1024*1024)
        print(f"\n  Done! ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"\n  FAILED: {e}")
        if filepath.exists():
            filepath.unlink()
        return False


def download_socrata_csv(base_url, query_params, filepath, description, max_rows=None):
    """Download from a Socrata API endpoint."""
    print(f"\n{'='*60}")
    print(f"Downloading: {description}")
    print(f"{'='*60}")
    
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024*1024)
        print(f"  Already exists ({size_mb:.1f} MB) — skipping")
        return True
    
    try:
        # Socrata paginates at 50k by default; we'll batch
        limit = max_rows or 1_000_000
        batch = 50000
        offset = 0
        all_dfs = []
        
        while offset < limit:
            this_limit = min(batch, limit - offset)
            params = f"$limit={this_limit}&$offset={offset}"
            if query_params:
                params += f"&{query_params}"
            
            url = f"{base_url}?{params}"
            print(f"  Fetching rows {offset:,}-{offset+this_limit:,}...", end=' ', flush=True)
            
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            
            df = pd.read_csv(StringIO(resp.text))
            print(f"got {len(df):,} rows")
            
            if len(df) == 0:
                break
            
            all_dfs.append(df)
            offset += this_limit
            
            if len(df) < this_limit:
                break  # no more data
        
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(filepath, index=False)
            size_mb = filepath.stat().st_size / (1024*1024)
            print(f"  Saved {len(combined):,} total rows ({size_mb:.1f} MB)")
            return True
        else:
            print("  No data returned!")
            return False
    
    except Exception as e:
        print(f"  FAILED: {e}")
        if filepath.exists():
            filepath.unlink()
        return False


# ============================================================
# 1. PHILADELPHIA
# ============================================================

def download_philadelphia():
    fp = DATA_DIR / "philadelphia_properties.parquet"
    if fp.exists():
        print(f"\n  Philadelphia: already exists — skipping")
        return True
    
    # Try CSV first, convert to parquet
    csv_fp = DATA_DIR / "philadelphia_properties.csv"
    
    # OpenDataPhilly OPA Properties
    # Socrata endpoint
    success = download_socrata_csv(
        base_url="https://phl.carto.com/api/v2/sql",
        query_params=None,
        filepath=csv_fp,
        description="Philadelphia (OPA Properties)",
    )
    
    if not success:
        # Alternative: direct Socrata API
        print("  Trying alternative source...")
        url = ("https://data.phila.gov/resource/j4iy-fmzp.csv"
               "?$limit=600000"
               "&$select=lat,lng,market_value,category_code_description,total_livable_area")
        success = download_file(url, csv_fp, "Philadelphia (Socrata API)")
    
    if not success:
        # Final fallback: direct bulk download
        url = "https://opendata.arcgis.com/api/v3/datasets/1c57dd1f3ff4457099a1880e2f20ab3f_0/downloads/data?format=csv&spatialRefId=4326"
        success = download_file(url, csv_fp, "Philadelphia (ArcGIS bulk)")
    
    if success and csv_fp.exists():
        try:
            df = pd.read_csv(csv_fp, low_memory=False)
            df.to_parquet(fp, index=False)
            print(f"  Converted to parquet: {fp}")
            return True
        except Exception as e:
            print(f"  Parquet conversion failed: {e}")
            # Rename CSV so the notebook can find it
            csv_fp.rename(DATA_DIR / "philadelphia_properties.csv")
            return True
    
    print("\n  *** MANUAL DOWNLOAD NEEDED ***")
    print("  Go to: https://opendataphilly.org/datasets/opa-property-assessments/")
    print("  Download CSV, save as: data/external/philadelphia_properties.csv")
    return False


# ============================================================
# 2. WASHINGTON D.C.
# ============================================================

def download_dc():
    fp = DATA_DIR / "dc_properties.parquet"
    if fp.exists():
        print(f"\n  D.C.: already exists — skipping")
        return True
    
    csv_fp = DATA_DIR / "dc_properties.csv"
    
    # DC Open Data — Integrated Tax System Public Extract
    success = download_socrata_csv(
        base_url="https://opendata.dc.gov/api/v2/sql",
        query_params=None,
        filepath=csv_fp,
        description="Washington D.C. (ITS Public Extract)",
    )
    
    if not success:
        # Alternative: Socrata endpoint
        url = ("https://opendata.dc.gov/resource/f6b5-553b.csv"
               "?$limit=300000"
               "&$select=LATITUDE,LONGITUDE,LANDAREA,AYB,ASSESSMENT_NBHD,"
               "PRICE,LAND,BLDG,USECODE,NUM_UNITS,GBA")
        success = download_file(url, csv_fp, "D.C. (Socrata)")
    
    if not success:
        # Final: direct download
        url = "https://opendata.dc.gov/api/v3/datasets/f6b5-553b/downloads/data?format=csv&spatialRefId=4326"
        success = download_file(url, csv_fp, "D.C. (ArcGIS bulk)")
    
    if success and csv_fp.exists():
        try:
            df = pd.read_csv(csv_fp, low_memory=False)
            df.to_parquet(fp, index=False)
            print(f"  Converted to parquet: {fp}")
            return True
        except:
            return True
    
    print("\n  *** MANUAL DOWNLOAD NEEDED ***")
    print("  Go to: https://opendata.dc.gov/datasets/integrated-tax-system-public-extract")
    print("  Download CSV, save as: data/external/dc_properties.csv")
    return False


# ============================================================
# 3. NEW YORK CITY
# ============================================================

def download_nyc():
    fp = DATA_DIR / "nyc_pluto.csv"
    if fp.exists():
        print(f"\n  NYC: already exists — skipping")
        return True
    
    # NYC PLUTO via Socrata API (select only needed columns for speed)
    success = download_socrata_csv(
        base_url="https://data.cityofnewyork.us/resource/64uk-42ks.csv",
        query_params=(
            "$select=bbl,latitude,longitude,assessland,assesstot,"
            "bldgclass,landuse,unitsres,lotarea,bldgarea,numfloors"
            "&$where=assesstot > 0 AND latitude IS NOT NULL"
        ),
        filepath=fp,
        description="New York City (PLUTO)",
        max_rows=900000,
    )
    
    if not success:
        # Alternative: full file download (larger, ~200MB)
        print("  Trying full PLUTO download (this is ~200MB)...")
        url = "https://data.cityofnewyork.us/api/views/64uk-42ks/rows.csv?accessType=DOWNLOAD"
        success = download_file(url, fp, "NYC PLUTO (full download)")
    
    if not success:
        print("\n  *** MANUAL DOWNLOAD NEEDED ***")
        print("  Go to: https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-PLUTO-/64uk-42ks")
        print("  Click Export → CSV")
        print("  Save as: data/external/nyc_pluto.csv")
    
    return success


# ============================================================
# 4. BOSTON
# ============================================================

def download_boston():
    fp = DATA_DIR / "boston_assessment.csv"
    if fp.exists():
        print(f"\n  Boston: already exists — skipping")
        return True
    
    # Try FY2026, then FY2025
    urls = [
        ("https://data.boston.gov/dataset/e02c44d2-3c64-459c-8fe2-e1ce5f38a035/"
         "resource/ee73430d-96c0-423e-ad21-c4cfb54c8961/download/"
         "fy2026-property-assessment-data_12_23_2025.csv",
         "Boston FY2026"),
        ("https://data.boston.gov/dataset/e02c44d2-3c64-459c-8fe2-e1ce5f38a035/"
         "resource/6b7e460e-33f6-4e61-80bc-1bef2e73ac54/download/"
         "fy2025-property-assessment-data_12_30_2024.csv",
         "Boston FY2025"),
    ]
    
    for url, desc in urls:
        if download_file(url, fp, desc):
            return True
    
    print("\n  *** MANUAL DOWNLOAD NEEDED ***")
    print("  Go to: https://data.boston.gov/dataset/property-assessment")
    print("  Download the latest FY CSV")
    print("  Save as: data/external/boston_assessment.csv")
    return False


# ============================================================
# 5. DETROIT
# ============================================================

def download_detroit():
    fp = DATA_DIR / "detroit_parcels.csv"
    if fp.exists():
        print(f"\n  Detroit: already exists — skipping")
        return True
    
    # Detroit Open Data — Parcels dataset
    success = download_socrata_csv(
        base_url="https://data.detroitmi.gov/resource/evhk-jg2y.csv",
        query_params=(
            "$select=parcelno,assessed_value,taxable_value,propclass,"
            "total_sqft,total_floor_area,year_built,latitude,longitude"
            "&$where=assessed_value > 0 AND latitude IS NOT NULL"
        ),
        filepath=fp,
        description="Detroit (Parcels)",
        max_rows=400000,
    )
    
    if not success:
        # Try without column selection
        success = download_socrata_csv(
            base_url="https://data.detroitmi.gov/resource/evhk-jg2y.csv",
            query_params="$where=assessed_value > 0",
            filepath=fp,
            description="Detroit (Parcels — all columns)",
            max_rows=400000,
        )
    
    if not success:
        print("\n  *** MANUAL DOWNLOAD NEEDED ***")
        print("  Go to: https://data.detroitmi.gov/")
        print("  Search for 'Parcels', click Export → CSV")
        print("  Save as: data/external/detroit_parcels.csv")
    
    return success


# ============================================================
# 6. SEATTLE / KING COUNTY
# ============================================================

def download_seattle():
    fp = DATA_DIR / "seattle_parcels.csv"
    if fp.exists():
        print(f"\n  Seattle: already exists — skipping")
        return True
    
    print(f"\n{'='*60}")
    print("Seattle / King County — SEMI-MANUAL DOWNLOAD")
    print(f"{'='*60}")
    print()
    print("King County requires accepting terms on their website.")
    print()
    print("OPTION A (recommended): Download + merge two files")
    print("-" * 50)
    print("1. Go to: https://info.kingcounty.gov/assessor/datadownload/default.aspx")
    print("2. Check the box to accept terms")
    print("3. Download 'Real Property Account' → EXTR_RPAcct.csv")
    print("4. Download 'Parcel' → EXTR_Parcel.csv")
    print("5. Place both files in data/external/")
    print("6. Run this merge code in Python:")
    print()
    print("   import pandas as pd")
    print("   acct = pd.read_csv('data/external/EXTR_RPAcct.csv', low_memory=False)")
    print("   parcel = pd.read_csv('data/external/EXTR_Parcel.csv', low_memory=False)")
    print("   # Keep relevant columns from parcel")
    print("   pcols = [c for c in ['Major','Minor','PropType','Range','Township',")
    print("            'Latitude','Longitude'] if c in parcel.columns]")
    print("   merged = acct.merge(parcel[pcols], on=['Major','Minor'], how='left')")
    print("   merged.to_csv('data/external/seattle_parcels.csv', index=False)")
    print("   print(f'Saved {len(merged)} rows')")
    print()
    print("OPTION B: Skip Seattle")
    print("-" * 50)
    print("The notebook will run with 5 comparison metros instead of 6.")
    print("This is still a strong analysis — 5 metros gives good coverage.")
    print()
    
    # Check if raw files exist for auto-merge
    acct_fp = DATA_DIR / "EXTR_RPAcct.csv"
    parcel_fp = DATA_DIR / "EXTR_Parcel.csv"
    
    if acct_fp.exists() and parcel_fp.exists():
        print("Found raw King County files — merging automatically...")
        try:
            acct = pd.read_csv(acct_fp, low_memory=False)
            parcel = pd.read_csv(parcel_fp, low_memory=False)
            pcols = [c for c in ['Major','Minor','PropType','Latitude','Longitude']
                     if c in parcel.columns]
            merged = acct.merge(parcel[pcols], on=['Major','Minor'], how='left')
            merged.to_csv(fp, index=False)
            print(f"  Saved {len(merged):,} rows to {fp}")
            return True
        except Exception as e:
            print(f"  Merge failed: {e}")
    
    return False


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  METRO DATA DOWNLOADER")
    print("  Downloads property assessment data for 6 comparison metros")
    print("=" * 60)
    
    results = {}
    
    for name, func in [
        ("Philadelphia, PA", download_philadelphia),
        ("Washington, D.C.", download_dc),
        ("New York City, NY", download_nyc),
        ("Boston, MA", download_boston),
        ("Detroit, MI", download_detroit),
        ("Seattle, WA", download_seattle),
    ]:
        results[name] = func()
    
    # Summary
    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "✓ Ready" if ok else "✗ NEEDS MANUAL DOWNLOAD"
        print(f"  {name:25s} {status}")
    
    ready = sum(1 for v in results.values() if v)
    print(f"\n  {ready}/6 metros ready")
    
    if ready < 6:
        print("\n  For any failed downloads, follow the manual instructions above.")
        print("  Then re-run this script — it will skip already-downloaded files.")
    
    print(f"\n  All files saved to: {DATA_DIR.resolve()}/")
    print("  Now open cross_metro_comparison.ipynb and run all cells.")


if __name__ == "__main__":
    main()
