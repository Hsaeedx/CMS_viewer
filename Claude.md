This workspace is meant for research projects related to the CMS (Medicare claims) database. There are multiple projects being done concurrently, including:
    - TORS vs CT/CRT for Oropharyngeal SCC
    - Outcomes in stroke patients who use SLPs vs not
    - IO Hospitalizations and Hospice in mucosal HNC

If on windows, only use powershell for this project. If you use bash, it must be simply to access powershell.

Read the .env file to understand file locations. The database is on an external drive.

You MUST read the documentation of the different tables and columns at F:\CMS\Documentation

When running /compact, be sure to reiterate what project you have been working on and what data structures and design elements are relevant to what I am doing.

Any change/update to the project should be reflected in the project's pipeline

For all figures, tables, and spec sheets, use the following color scheme:
    Primary:
        Scarlet (PMS 200)	#ba0c2f	$scarlet	
        Gray (PMS 429)	#a7b1b7	$gray	
        White	#ffffff	$white

    Secondary
        Scarlet Dark 40	#70071c	$scarlet-dark-40	
        Scarlet Dark 60	#4a0513	$scarlet-dark-60


When running DuckDB queries, prefer using powershell directy over invoking python if there is no secondary use of the python script
    - i.e. Get-Content F:\CMS\queries\opscc_outcomes.sql | duckdb F:\CMS\cms_data.duckdb

Check cms_data.duckdb to understand the scheme of my data.

Look at diagnoses.json to understand how diagnoses are defined.

Look at headers.json to understand what data is available in the cms_data duckdb.

Look at build_cohort.py to see how cohorts are built.