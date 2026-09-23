# Bird Image Retrieval (CBIR)

Hệ thống web tìm ảnh chim tương tự bằng **đặc trưng thủ công**. Phiên bản `v4-color` không dùng CNN, ViT, CLIP hay deep embedding cho retrieval. Ba nhóm đặc trưng được lưu và so sánh độc lập trước khi fusion.

## 1. Kiến trúc

```text
Browser / REST client
        |
        v
FastAPI + Jinja2  ---> Query preprocessing (YOLO-Seg)
        |                       |
        |                       v
        |             Color 504D / Texture 74D / Shape 1833D
        |                       |
        v                       v
PostgreSQL <-------- exhaustive per-group similarity
        |
        v
Top-K + component similarities
```

- `app/services/preprocessing.py`: đọc ảnh Unicode trên Windows, YOLO-Seg cho query thô, kiểm tra/clean mask, crop–resize–pad 384×384.
- `app/services/orientation.py`: hook orientation. V1 cố ý để `ORIENTATION_MODE=none`; mask không đủ tin cậy để suy ra đầu/mỏ. UI có tùy chọn lật ngang thủ công.
- `color_features.py`, `texture_features.py`, `gabor_features.py`, `shape_features.py`, `hog_features.py`: ba nhóm descriptor độc lập; texture có Gabor đa tỉ lệ/đa hướng và shape có HOG đa vùng.
- `normalization.py`: StandardScaler dành riêng cho 10 GLCM, 36 Gabor và 5 shape scalar, fit trên dataset rồi lưu Joblib.
- `similarity.py`: metric theo nhóm, khoảng cách neo tại 0 và scale bằng p95 để chống outlier mà không làm hòa điểm Top-K; p5 được giữ làm thống kê chẩn đoán. Fusion có điều chỉnh theo mask quality.
- `indexing.py`: index recursive, ghi ảnh chuẩn hóa riêng, không sửa ảnh nguồn, upsert theo đường dẫn tuyệt đối.
- `routes/`: web và REST API.

## 2. Database schema

PostgreSQL được dùng trực tiếp qua SQLAlchemy 2.x.

```text
images
  id PK
  file_name
  file_path UNIQUE
  processed_path
  species NULL
  width, height
  created_at

image_features
  id PK
  image_id UNIQUE FK -> images.id ON DELETE CASCADE
  color_vector REAL[]
  texture_vector REAL[]
  shape_vector REAL[]
  feature_version
  created_at
```

## 3. Đặc trưng v4-color

| Nhóm | Thành phần | Chiều | Metric |
|---|---|---:|---|
| Color–Spatial | HSV tròn 18×4×4 + 3 vùng × HSV 12×3×2, có trọng số saturation | 504 | Chi-square |
| Texture | uniform LBP (8,1) + (16,2), masked GLCM 5 property × mean/std | 38 | Chi-square + Euclidean |
| Multi-scale Gabor | 3 bước sóng × 6 hướng × mean/std đáp ứng foreground | 36 | Euclidean |
| Robust Shape | occupancy grid 8×8 + aspect ratio/eccentricity/circularity/solidity/extent | 69 | Euclidean |
| Multi-region HOG | ảnh xám có mask 128×128, cell 16×16, 49 block 2×2 chồng lấn, 9 hướng | 1764 | Euclidean |

Tổng là 2411 chiều nhưng **không nối toàn bộ để tính một Euclidean duy nhất**. Hue được làm mượt tuần hoàn để nối vùng đỏ 0/179 và pixel ít bão hòa được giảm trọng số. GLCM tự xây ma trận từ những cặp mà cả hai pixel nằm trong mask; Gabor chỉ thống kê đáp ứng foreground và HOG xóa nền trước khi tính gradient.

## 4. Cài đặt trên Windows

Yêu cầu Python 3.11+ và PostgreSQL đang chạy.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Tạo database bằng `psql` hoặc pgAdmin:

```sql
CREATE DATABASE bird_cbir;
```

Sửa `DATABASE_URL` trong `.env` theo mật khẩu local. Đặt model segmentation tại `models/yolo11n-seg.pt`, hoặc sửa `YOLO_MODEL` trỏ tới model có sẵn. Project tham khảo hiện có model tại `D:\PROJECT\TIEN XLA V1\yolo11n-seg.pt`.

Khởi tạo bảng:

```powershell
python scripts/init_db.py
```

## 5. Index ba dataset hiện có

Ba folder chứa PNG RGBA 384×384 và mask alpha, nên phải bật `--already-preprocessed` để không chạy YOLO lại. Lần kiểm kê đầu có 1.469 ảnh; tại thời điểm smoke test còn 1.468 ảnh (file `NA_TACH_NEN/114.Black_throated_Sparrow/Black_Throated_Sparrow_0007_106999.png` được report cũ ghi thành công nhưng hiện không còn trên đĩa).

```powershell
python scripts/index_dataset.py --dataset `
  "D:\PROJECT\ANH TACH NEN\NA_TACH_NEN" `
  "D:\PROJECT\ANH TACH NEN\ĐỨC_TACH_NEN" `
  "D:\PROJECT\ANH TACH NEN\CUONG_TACH_NEN" `
  --already-preprocessed
```

Script duyệt recursive, lấy `species` từ folder cha, tiếp tục khi một ảnh lỗi và hiển thị progress. Ảnh chuẩn hóa được ghi vào `app/static/dataset`; ảnh nguồn không bị ghi đè. Chạy lại sẽ upsert thay vì nhân đôi record, đồng thời fit lại scaler và thống kê p5/p95 trên toàn bộ input của lần chạy.

Với dataset khác chưa tách nền, bỏ cờ `--already-preprocessed` để dùng YOLO-Seg. `scripts/recompute_features.py` nhận cùng tham số khi cần tính lại feature version hiện tại.

## 6. Chạy web và truy vấn

```powershell
uvicorn app.main:app --reload
```

Mở <http://127.0.0.1:8000>, chọn ảnh, Top K (5/10/20), tùy chọn lật ngang rồi bấm **Tìm ảnh chim tương tự**. Hệ thống có một chế độ CBIR duy nhất: xếp hạng toàn bộ ảnh theo độ tương đồng thị giác, không lọc hoặc ép kết quả theo nhãn species.

API:

```text
POST /api/search       multipart: image, top_k, flip_horizontal
GET  /api/images/{id}  ảnh PNG đã chuẩn hóa
GET  /api/health
GET  /api/stats
```

Ví dụ curl:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/search `
  -F "image=@bird.jpg" -F "top_k=10" -F "flip_horizontal=false"
```

## 7. Test, debug và đánh giá

```powershell
pytest
python scripts/evaluate_features.py
```

Evaluation thực hiện leave-one-out và in P@1, P@5, P@10, R@10, mAP@10 cho Color, Texture, Shape và các tổ hợp ablation; chạy đầy đủ là O(N²).

`scripts/evaluate_features.py` còn chạy ablation sâu từng thành phần và phép bỏ lần lượt một thành phần khỏi toàn pipeline. Trọng số ablation sâu chỉ được chọn trên calibration; kết quả cuối được đo trên test và lưu tại `models/ablation_report.json`.

Khảo sát PCA riêng cho HOG, gồm cả biến thể có/không StandardScaler, chạy bằng `python scripts/evaluate_hog_pca.py`. PCA chỉ fit trên calibration; kết quả test và thời gian tính khoảng cách được lưu tại `models/hog_pca_report.json`.

Khảo sát species centroid/prototype và soft reranking chạy bằng `python scripts/evaluate_species_prototypes.py`. Centroid của đúng loài được tính leave-one-out cho từng query; alpha chỉ được chọn trên calibration. Báo cáo được lưu tại `models/species_prototype_report.json`.

Khảo sát weighted kNN theo loài chạy bằng `python scripts/evaluate_weighted_knn.py`. Script thử các cấu hình N/p/beta, chọn cấu hình theo P@1 trên calibration và chỉ sau đó đánh giá test. Báo cáo được lưu tại `models/weighted_knn_report.json`.

Tuning trọng số có thể tái lập bằng:

```powershell
python scripts/tune_weights.py
python scripts/recompute_distance_stats.py
```

Thí nghiệm seed 42 chia query theo từng species thành 675 calibration và 772 test. So với v3-gabor, v4-color cải thiện trên test độc lập: P@1 `0.2552 -> 0.2707`, P@5 `0.1645 -> 0.1806`, P@10 `0.1289 -> 0.1364`, Recall@10 `0.1405 -> 0.1510`, mAP@10 `0.1023 -> 0.1127`. Báo cáo đầy đủ nằm ở `models/weight_tuning_report.json`.

Đặt `DEBUG_FEATURES=true` để mỗi query lưu `original`, `mask`, `cropped`, `processed_384`, visualization grid 8×8 và ba vùng màu vào `debug_outputs/<query-id>/`. Response còn trả mask quality và effective weights.

## 8. Quyết định và giới hạn

- Search v4-color quét toàn bộ vector trong PostgreSQL. Cách này rõ ràng và phù hợp quy mô vài nghìn ảnh, nhưng chưa dành cho hàng triệu record.
- Mask alpha dataset được tin là kết quả segmentation; vẫn morphology nhẹ, validate và chuẩn hóa lại. Fallback nền hằng chỉ dùng khi người vận hành chủ động bật `--already-preprocessed` cho ảnh không alpha.
- Border contact làm giảm quality score; mask rỗng, diện tích dưới 1% hoặc bbox quá nhỏ bị loại.
- YOLO chọn candidate chim bằng confidence nhân căn bậc hai tỉ lệ diện tích, cân bằng độ tin cậy với object quá nhỏ.
- Không tự suy đoán đầu/đuôi từ silhouette. Người dùng kiểm tra hai hướng bằng checkbox flip.
- Scaler và distance statistics phải khớp database. Khi đổi dataset/thuật toán/trọng số cấp thấp, hãy index lại toàn bộ tập dùng cho search.
- Project không chứa sẵn binary model YOLO và không chứa mật khẩu PostgreSQL.
