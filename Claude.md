This workspace is meant for research projects related to the CMS (Medicare claims) database. There are multiple projects being done concurrently, including:
    - TORS vs CT/CRT for Oropharyngeal SCC
    - Outcomes in stroke patients who use SLPs vs not

Only use powershell for this project. If you use bash, it must be simply to access powershell.

All data files and research project specific files are located at F:\CMS
All MAIN project files are lcoated at C:\Users\hsaee\Desktop\CMS_viewer

Documentation of the different tables and columns can be found at F:\CMS\Documentation

When running /compact, be sure to reiterate what project you have been working on and what data structures and design elements are relevant to what I am doing.

When running DuckDB queries, prefer using powershell directy over invoking python if there is no secondary use of the python script
    - i.e. Get-Content F:\CMS\queries\opscc_outcomes.sql | duckdb F:\CMS\cms_data.duckdb

Check cms_data.duckdb to understand the scheme of my data.

Look at diagnoses.json to understand how diagnoses are defined.

Look at headers.json to understand what data is available in the cms_data duckdb.

Look at build_cohort.py to see how cohorts are built.