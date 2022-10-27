SELECT world_size, plain_text, batch_size, ttp, acc, inference_time_s
FROM `PrivateInference`
WHERE world_size = 3 AND ttp = 0
ORDER BY `inference_time_s`