# Data provenance checklist

For each dataset, record:

- Dataset name and domain
- Original source URL
- License
- Retrieval date
- File checksum (SHA-256)
- Original temporal frequency
- Aggregation procedure to weekly frequency
- Date range
- Missing-value treatment
- Duplicate handling
- Train/test split timestamp
- Window sizes considered and selected

Do not commit restricted raw data. Commit download/preparation scripts and checksums instead.
