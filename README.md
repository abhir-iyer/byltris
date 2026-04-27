# Byltris

Consumer financial stress intelligence built on public regulatory data.

Three analytical modules:
- **Early Warning** — FDIC call report distress prediction (Cox PHM + XGBoost)
- **Complaint Intelligence** — CFPB narrative NLP (BERTopic)
- **Fair Lending** — HMDA disparity decomposition (two-stage regression)

## Stack
Python · XGBoost · lifelines · BERTopic · Prophet · FastAPI · Next.js

## Data Sources
- FDIC: https://banks.data.fdic.gov/api
- CFPB: https://www.consumerfinance.gov/data-research/consumer-complaints/
- HMDA: https://ffiec.cfpb.gov/data-browser/
- FRED: https://fred.stlouisfed.org/
