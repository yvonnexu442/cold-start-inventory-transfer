# Paper dataset placement

Raw datasets are obtained from their original sources, remain outside version
control, and are not redistributed by this repository.

## MAN and BRAF service-parts data

Place the `Spare-Part-Demand-Forecasting-main.zip` source archive under
`data/raw/spdf/`. The versioned parser reads the MAN and BRAF files used by the
service-parts evaluation. Retain the original archive so its checksum can be
matched to the frozen experiment record.

The source benchmark identifies these datasets only as MAN and BRAF and does
not document formal expansions. The manuscript therefore retains the labels as
source identifiers.

## Online Retail II (UCI Machine Learning Repository)

Download Online Retail II from the UCI Machine Learning Repository and place
the archive and workbook at:

```text
data/raw/uci_online_retail_ii/online+retail+ii.zip
data/raw/uci_online_retail_ii/online_retail_II.xlsx
```

The workbook is read by `scripts/run_uci_online_retail_ii_external.py` using
`configs/uci_online_retail_ii_external.yaml`.

## Corporación Favorita

Accept the terms and obtain the archive from the official Kaggle competition:
`https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting`.
Preserve it at:

```text
data/raw/favorita/favorita-grocery-sales-forecasting.zip
```

Extract `items.csv`, `stores.csv`, and `train.csv` under
`data/raw/favorita/extracted_v1/csv/`. Verify their hashes against the
legacy-compatible manifest path
`outputs/ai_darld_v3/favorita_confirmation_v1/run_manifest.json`. Raw
competition data must not be committed or redistributed.

Historical dataset-discovery pilots for other datasets are not part of this
public reproduction snapshot.
