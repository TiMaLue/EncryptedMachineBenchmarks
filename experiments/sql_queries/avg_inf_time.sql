SELECT avg(inference_time_s)
FROM `PrivateInferenceImageClsEnc`
WHERE inference_time_s is not null