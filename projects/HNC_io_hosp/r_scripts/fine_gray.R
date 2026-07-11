#!/usr/bin/env Rscript
# fine_gray.R
# Fine-Gray subdistribution hazard model via cmprsk::crr().
#
# Reads a CSV with columns:
#   time_to_event, event_type (1=hospice, 2=death w/o hospice), covariates ...
# Writes results as CSV: term, log_shr, se, shr, ci_lo, ci_hi, p
#
# Usage: Rscript fine_gray.R <input_csv> <formula_covariate_names_comma_sep> <output_csv>

suppressPackageStartupMessages(library(cmprsk))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
    stop("Usage: Rscript fine_gray.R <input_csv> <covariate_names_csv> <output_csv>")
}
input_csv  <- args[1]
covar_str  <- args[2]
output_csv <- args[3]

df <- read.csv(input_csv, stringsAsFactors = FALSE)

# Split covariate names
covar_names <- strsplit(covar_str, ",", fixed = TRUE)[[1]]
cat("Covariates provided:", covar_names, "\n")

# Build design matrix. Explicit factor reference levels (first = reference).
# Must match make_regression.py REF_ORDER dict.
REF_LEVELS <- list(
    sex                    = c("Male", "Female"),
    race_collapsed         = c("White", "Black", "Hispanic", "Other/Unknown"),
    urban_rural            = c("Metro", "Non-metro", "Unknown"),
    census_region          = c("South", "Northeast", "Midwest", "West", "Unknown"),
    subsite_category       = c("Hypopharynx", "Larynx", "Oral Cavity", "Oropharynx"),
    io_agent               = c("pembrolizumab", "nivolumab", "both"),
    io_regimen             = c("ICI monotherapy", "chemo-ICI"),
    primary_curative_type  = c("radiation", "surgery")
)

cov_df <- df[, covar_names, drop = FALSE]
for (col in covar_names) {
    if (col %in% names(REF_LEVELS)) {
        levels_spec <- REF_LEVELS[[col]]
        # Guard: only keep levels that actually appear in the data
        levels_present <- intersect(levels_spec, unique(cov_df[[col]]))
        cov_df[[col]] <- factor(cov_df[[col]], levels = levels_present)
    } else if (is.character(cov_df[[col]])) {
        # Fallback for any un-anticipated categorical
        cov_df[[col]] <- factor(cov_df[[col]])
    }
}
# model.matrix(~ .) creates dummy variables using the first level as reference;
# drop the intercept column (index 1) to get the design matrix in Fine-Gray form
X <- model.matrix(~ ., data = cov_df)[, -1, drop = FALSE]

cat("Design matrix cols:", ncol(X), "\n")
cat("Head:\n"); print(head(X, 2))

# Fit Fine-Gray: event=1 is the event of interest, event=2 is the competing event
fit <- crr(ftime  = df$time_to_event,
           fstatus = df$event_type,
           cov1   = X,
           failcode = 1,
           cencode  = 0,
           maxiter = 200)

# Extract coefficients + CIs
coef_summary <- summary(fit)
tab <- coef_summary$coef
# tab columns: coef, exp(coef), se(coef), z, p-value
# CI: exp(coef +/- 1.96*se(coef))

out <- data.frame(
    term    = rownames(tab),
    log_shr = as.numeric(tab[, "coef"]),
    se      = as.numeric(tab[, "se(coef)"]),
    shr     = as.numeric(tab[, "exp(coef)"]),
    z       = as.numeric(tab[, "z"]),
    p       = as.numeric(tab[, "p-value"]),
    stringsAsFactors = FALSE
)
out$ci_lo <- exp(out$log_shr - 1.96 * out$se)
out$ci_hi <- exp(out$log_shr + 1.96 * out$se)

# Meta row
n_events_1 <- sum(df$event_type == 1)
n_events_2 <- sum(df$event_type == 2)
n_total    <- nrow(df)

write.csv(out, output_csv, row.names = FALSE)
cat(sprintf("Fine-Gray fit: n=%d, event1=%d, event2=%d\n",
            n_total, n_events_1, n_events_2))
cat(sprintf("Wrote %s\n", output_csv))
