# Data Overview

- **Source file**: `data/startups_raw.csv`
- **Number of rows**: 66368
- **Number of columns**: 14

## Dataset schema

| Column | Inferred type | Non-null | Null | % missing | Examples |
|--------|-------------|----------|------|------------|----------|
| permalink | string | 66368 | 0 | 0.00% | /organization/-fame, /organization/-qounter, /organization/-the-one-of-them-inc-, /organization/0-6-com, /organization/004-technologies |
| name | string | 66367 | 1 | 0.00% | #fame, :Qounter, (THE) ONE of THEM,Inc., 0-6.com, 004 Technologies |
| homepage_url | string | 61310 | 5058 | 7.62% | http://livfame.com, http://www.qounter.com, http://oneofthem.jp, http://www.0-6.com, http://004gmbh.de/en/004-interact |
| category_list | string | 63220 | 3148 | 4.74% | Media, Application Platforms|Real Time|Social Network Media, Apps|Games|Mobile, Curated Web, Software |
| funding_total_usd | string | 66368 | 0 | 0.00% | 10000000, 700000, 3406878, 2000000, - |
| status | string | 66368 | 0 | 0.00% | operating, operating, operating, operating, operating |
| country_code | string | 59410 | 6958 | 10.48% | IND, USA, CHN, USA, HKG |
| state_code | string | 57821 | 8547 | 12.88% | 16, DE, 22, IL, BC |
| region | string | 58338 | 8030 | 12.10% | Mumbai, DE - Other, Beijing, Springfield, Illinois, Hong Kong |
| city | string | 58340 | 8028 | 12.10% | Mumbai, Delaware City, Beijing, Champaign, Hong Kong |
| funding_rounds | int | 66368 | 0 | 0.00% | 1, 2, 1, 1, 1 |
| founded_at | date | 51147 | 15221 | 22.93% | 2014-09-04, 2007-01-01, 2010-01-01, 1997-01-01, 2011-01-01 |
| first_funding_at | date | 66344 | 24 | 0.04% | 2015-01-05, 2014-03-01, 2014-01-30, 2008-03-19, 2014-07-24 |
| last_funding_at | date | 66368 | 0 | 0.00% | 2015-01-05, 2014-10-14, 2014-01-30, 2008-03-19, 2014-07-24 |

## Missing values per column

- **permalink**: 0 missing (0.00%)
- **name**: 1 missing (0.00%)
- **homepage_url**: 5058 missing (7.62%)
- **category_list**: 3148 missing (4.74%)
- **funding_total_usd**: 0 missing (0.00%)
- **status**: 0 missing (0.00%)
- **country_code**: 6958 missing (10.48%)
- **state_code**: 8547 missing (12.88%)
- **region**: 8030 missing (12.10%)
- **city**: 8028 missing (12.10%)
- **funding_rounds**: 0 missing (0.00%)
- **founded_at**: 15221 missing (22.93%)
- **first_funding_at**: 24 missing (0.04%)
- **last_funding_at**: 0 missing (0.00%)

## Status distribution (success / failure)

| Status | Count | % |
|--------|--------|---|
| operating | 53034 | 79.91% |
| closed | 6238 | 9.40% |
| acquired | 5549 | 8.36% |
| ipo | 1547 | 2.33% |

## Warnings and points of attention

- **Duplicates**: no obvious duplicates detected based on the identifier columns used.
- **Date columns to double-check**: founded_at, first_funding_at, last_funding_at