SELECT pi.world_size, pi.plain_text, pi.batch_size,
 pi.acc as acc_1, pi.inference_time_s inference_time_s_1,
 pi2.acc as acc_2, pi2.inference_time_s inference_time_s_2,
 pi.acc - pi2.acc as acc_diff,
 pi2.inference_time_s/pi.inference_time_s as speedup
FROM `PrivateInference` as pi, `PrivateInference_2` as pi2
WHERE pi.world_size = pi2.world_size AND
 pi.plain_text = pi2.plaintext AND
 pi.batch_size = pi2.batch_size AND
 pi.plain_text = 0
ORDER BY pi.world_size, pi.plain_text, pi.batch_size;



