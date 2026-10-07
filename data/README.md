# Local data policy

Raw data is never committed. Download every dataset manually and retain the source's original license and terms; this repository does not redistribute or relicense it. Generated interim and processed data also stays local unless a later release is intentional, documented, and legally permitted.

The submitted-paper datasets are MAN/BRAF service parts, UCI Online Retail II,
and Corporación Favorita. Follow `docs/dataset_download_instructions.md` for
their repository-relative placement. SPDF source archives belong under
`data/raw/spdf/`; UCI files under `data/raw/uci_online_retail_ii/`; and Favorita
files under `data/raw/favorita/`.

Favorita competition terms prohibit redistribution, so neither the archive nor
extracted CSV files are committed. The run manifest records source-file hashes.
The SPDF repository's code license must not be assumed to grant redistribution
rights for its industrial datasets.

Initial repository validation does not require any dataset file.
