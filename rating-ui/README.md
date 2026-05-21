# DDI Rating UI

Static HTML/JS rating interface for the clinician-validation study. Raters
classify items as benign or malignant (with an optional "uncertain" flag),
ratings persist in `localStorage`, and the rater downloads a CSV at the end.
No backend, no database -- pure static deployment.

## 1. Build the item manifest + bundle images

From the project root:

```bash
python scripts/sample_clinician_items.py --ddi-root /path/to/ddi
python scripts/build_rating_manifest.py --ddi-root /path/to/ddi
```

The second script:
- copies the 100 sampled images into `rating-ui/images/`
- writes `rating-ui/sample.json` with `{items: [{item_id, image_path}, ...]}`

## 2. Deploy

### Vercel (recommended)

```bash
cd rating-ui
npx vercel deploy --prod
```

You'll get a public URL like `https://ddi-rate-xxxx.vercel.app`. Share with
raters; they each enter their rater_id on the landing page and proceed.

### Locally (development)

```bash
cd rating-ui
python -m http.server 8000
# open http://localhost:8000 in a browser
```

## Data flow

1. `sample.json` is the **only** input the UI reads. It contains `item_id`
   and `image_path` for each item -- no ground truth, no FST, no IRT
   difficulty. Raters are fully blind.
2. As a rater clicks through, their answers persist in `localStorage`
   keyed by `ddi.ratings.<rater_id>`. Refreshing or closing the tab does
   not lose progress.
3. At the end, the rater clicks "Download CSV" and emails the file to the
   study coordinator. Schema:
   ```
   rater_id,rater_order,item_id,rater_label,rater_uncertain,timestamp
   ```
4. The coordinator concatenates per-rater CSVs and feeds them into a
   downstream analysis that adds them as additional respondent rows in
   the Rasch fit.

## Privacy / data-use notes

- DDI images are sourced from Daneshjou et al. (2022) under their data
  use agreement. Hosting them on a public Vercel URL is technically
  available-to-anyone-with-the-URL. If your IRB requires non-public
  access, deploy to a Stanford-IP-restricted host or run locally and
  share the local URL over VPN.
- No PHI is in the rating UI itself. Rater identifiers are arbitrary
  strings; pick something de-identified (e.g., `resident-01`).
