from measure import measure_encoder
from category_encoders import SumEncoder

metrics = measure_encoder(SumEncoder, save_results=True)