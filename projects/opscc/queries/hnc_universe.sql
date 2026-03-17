DROP TABLE IF EXISTS hnc_universe;

CREATE TABLE hnc_universe AS
SELECT DSYSRTKY, first_hnc_date
FROM hnc_confirmed;