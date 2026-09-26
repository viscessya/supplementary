# Glossary: Indonesian names used in this repository

The scripts, logs, and some table columns keep the Indonesian names they had while the study was run, so that the
published files are exactly the files that produced the manuscript's numbers (see README, "Paths"). Every
user-facing document (README, SUPPLEMENTARY_MATERIAL, photographs/README) is in
English. This glossary translates the recurring terms.

## Words in file and folder names

| Indonesian | English |
|---|---|
| analisis | analysis |
| evaluasi | evaluation |
| hasil | results |
| uji | test |
| uji_mc | marching-cubes (extractor) test |
| sintetis | synthetic |
| hitung | compute |
| jalankan | run |
| gerbang | gate (pre-evaluation check) |
| verifikasi / kontribusi | verification / contribution claims |
| manifes / reproduksibilitas | manifest / reproducibility |
| konversi | conversion |
| susun | assemble (a figure) |
| galeri / pendorong | gallery / driver components (the four components that carry most of the gain) |
| peta / histogram / visibilitas | map / histogram / visibility |
| fitur_objek | per-object geometric features |
| per_objek | per object |
| ringkasan | summary |
| tabel3 | "Table 3" at the time of writing = Table 4 in the manuscript (and "Table 4" then = Table 3 now) |
| Kandidat A / B / C / D | four of the input-conditioning variants as named during the study: perspective projection / semi-metallic material / diagonal azimuths / gray background |
| potong_dan_hapus_latar | crop and remove background |
| segmentasi | segmentation |
| latar_asli / latar_putih | real background / white background |
| pompa | pump |
| transparan | transparent |

## Column names

`results/fitur_objek.csv` (reference-mesh features, produced by `analysis/analisis_fitur_objek.py`):

| Column | Meaning |
|---|---|
| objek, berkas | object id, file name |
| dim_mm, ukuran_maks_mm | bounding-box dimensions and largest dimension (mm) |
| segitiga_gt | reference triangle count |
| komponen | connected components (bodies) |
| tepi_batas, tepi_nonmanifold | boundary edges, non-manifold edges |
| euler, genus | Euler number, genus (closed meshes only) |
| rasio_aspek | largest / smallest dimension |
| kepipihan | flatness: smallest / largest dimension |
| isi_bbox | mesh volume / bounding-box volume |
| rasio_luas_thd_hull, rasio_volume_thd_hull | surface area and volume relative to the convex hull |
| luas_hadap_atas, luas_hadap_bawah, luas_hadap_samping | area share facing up (n_z > 0.5), down (n_z < -0.5), sideways |
| terlihat_elev{0..40} | visible surface share at that elevation (four rig views) |
| delta_terlihat_0_20 | visible share at 20 degrees minus at 0 degrees |
| luas_terlihat_sisi0 | area seen by the four side views at 0 degrees |
| luas_terlihat_atas | area seen by a straight top view |
| luas_atas_saja | area seen only from the top, not by the 0-degree side views |
| luas_tak_terlihat | area seen by neither |

`results/visibilitas_per_objek.csv` (from `analysis/hitung_visibilitas.py`):

| Column | Meaning |
|---|---|
| objek, elevasi | object id, elevation (degrees) |
| terlihat, tak_terlihat | visible and not-visible surface share |
| evidensi | area-weighted mean cosine of the best viewing angle, not-visible = 0 (the cosine-weighted variant in Section 4.2) |
| theta_median_terlihat | median best viewing angle of visible points (degrees) |
| luas_theta_a-b | surface share whose best viewing angle lies between a and b degrees |

`results/seed_n30_per_objek.csv` (three-seed summary, from `analysis/analisis_seed_n30.py`):

| Column | Meaning |
|---|---|
| dCD_s42, dCD_s123, dCD_s777, dCD_rerata | Chamfer improvement from 0 to 20 degrees per seed, and their mean |
| arah | direction across the three seeds: `3 membaik` = improves under all three, `3 memburuk` = worsens under all three, `campur` = mixed |
| sd_cd_0, sd_cd_20 | standard deviation of Chamfer distance across seeds at 0 and 20 degrees |
| rasio_efek_rentang | absolute mean effect divided by the larger between-seed range (max minus min) at 0 or 20 degrees |
| sumber | source of the seed-123/777 runs: `kanonik+seed42` = newly generated, `e4lama+seed42` = earlier replicate runs reused (Section 3.5) |

`results/uji_mc_runlog.csv` (extractor-comparison run log): waktu = time, lengan = arm (elevation), objek = object,
detik = seconds, sn_byte / mc_byte = file size of the surface-net / marching-cubes mesh, catatan = note.

## Words in logs and code comments

| Indonesian | English |
|---|---|
| objek | object (component) |
| elevasi | elevation |
| rerata / median | mean / median |
| membaik / memburuk / campur | improves / worsens / mixed |
| lolos / gagal | passes / fails |
| prediksi / terdaftar | prediction / registered (fixed in the written plan before generation) |
| terukur | measured subset |
| tanpa 4 pendorong | without the four driver components |
| pembanding | comparison / control analysis |
| permutasi | permutation (test) |
| catatan | note |
