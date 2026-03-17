DROP TABLE IF EXISTS opscc_universe;

CREATE TABLE opscc_universe AS
SELECT DISTINCT u.DSYSRTKY, u.first_hnc_date
FROM hnc_universe u
JOIN hnc_dx_raw d
  ON u.DSYSRTKY = d.DSYSRTKY
WHERE d.dx_prefix IN ('C01','C09','C10','C14');