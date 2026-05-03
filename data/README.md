# Data Schema

Place the main CSV file at:

```text
data/merged_data.csv
```

Required columns:

| Column | Description |
|---|---|
| `text` | Review text. |
| `label` | Binary label. `1` means fake review and `0` means truthful review. |

Optional metadata columns used by the full model:

| Group | Columns |
|---|---|
| User/discount | `vip`, `discount` |
| Ratings | `overall`, `taste`, `environment`, `service`, `ingredient` |
| Consumption/content | `consumption`, `food`, `picture` |
| Interaction | `like`, `response`, `interaction` |
| Time | `time` |
| City/grouping | `city`, `shop_city`, `region`, or `district_city` |

The loader also derives:

- `publish_hour`
- `is_weekend`
- `text_length`
- `exclamation_count`
- `has_picture`
- `is_extreme_rating`
- `rating_mean`
- `rating_var`
- `rating_dev`
- `has_consumption`
- `valid_post_hour`

Missing optional metadata columns are filled with zeros. For fair reproduction of the reported full-model results, use the complete metadata schema whenever possible.

Raw review data is not committed to this repository.
