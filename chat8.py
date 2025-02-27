import os
import pandas as pd
import pickle
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Establecer la clave de API de OpenAI

# Ruta para guardar la base de datos vectorial
VECTOR_DB_DIR = "vector_dbs"
METADATA_DIR = "vector_db_metadatas"

# Lista de IDs de documentos
document_ids = [
      2022, 2023, 2024, 2026, 2037, 2042, 2044, 2045, 2047, 2048, 2049, 2050, 2052, 2054, 2055, 2057, 2064, 2065, 2068, 2069, 2074, 2075, 2078, 2081, 2083, 2085, 2087, 2089, 2090, 2091, 2092, 2096, 2099, 2101, 2102, 2105, 2109, 2110, 2112, 2115, 2118, 2119, 2121, 2123, 2124, 2125, 2126, 2131, 2132, 2136, 2138, 2140, 2143, 2146, 2150, 2155, 2160, 2163, 2164, 2167, 2170, 2171, 2178, 2182, 2183, 2184, 2185, 2194, 2200, 2205, 2206, 2207, 2219, 2222, 2226, 2227, 2228, 2237, 2240, 2244, 2247, 2249, 2250, 2251, 2255, 2258, 2259, 2260, 2262, 2263, 2264, 2265, 2266, 2274, 2275, 2278, 2280, 2282, 2283, 2284, 2288, 2289, 2292, 2293, 2294, 2296, 2297, 2302, 2308, 2309, 2311, 2312, 2313, 2314, 2319, 2321, 2324, 2325, 2329, 2331, 2333, 2334, 2335, 2338, 2339, 2343, 2347, 2348, 2360, 2361, 2362, 2363, 2364, 2373, 2374, 2375, 2379, 2380, 2387, 2388, 2392, 2393, 2394, 2395, 2397, 2398, 2406, 2407, 2409, 2411, 2412, 2415, 2416, 2417, 2419, 2420, 2422, 2424, 2425, 2426, 2427, 2428, 2429, 2437, 2443, 2444, 2447, 2448, 2453, 2454, 2458, 2459, 2460, 2462, 2464, 2465, 2469, 2470, 2482, 2485, 2486, 2487, 2489, 2490, 2491, 2492, 2493, 2494, 2495, 2496, 2505, 2508, 2514, 2517, 2518, 2520, 2521, 2523, 2524, 2525, 2527, 2531, 2532, 2537, 2551, 2552, 2553, 2555, 2556, 2557, 2559, 2561, 2566, 2570, 2577, 2579, 2583, 2599, 2601, 2611, 2612, 2613, 2615, 2625, 2636, 2650, 2652, 2657, 2658, 2664, 2666, 2681, 2686, 2694, 2701, 2716, 2734, 2736, 2737, 2738, 2740, 2741, 2742, 2744, 2753, 2756, 2758, 2762, 2764, 2767, 2776, 2780, 2782, 2785, 2786, 2787, 2790, 2794, 2795, 2796, 2799, 2800, 2802, 2803, 2804, 2805, 2806, 2817, 2820, 2822, 2823, 2826, 2827, 2830, 2831, 2832, 2835, 2840, 2842, 2844, 2845, 2847, 2854, 2862, 2865, 2866, 2867, 2870, 2873, 2876, 2882, 2884, 2894, 2896, 2903, 2909, 2917, 2918, 2920, 2927, 2930, 2934, 2937, 2938, 2939, 2941, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2952, 2956, 2958, 2959, 2964, 2965, 2968, 2972, 2973, 2975, 2979, 2980, 2983, 2992, 2993, 2995, 2997, 2999, 3005, 3006, 3014, 3015, 3016, 3019, 3020, 3022, 3025, 3028, 3029, 3032, 3033, 3034, 3038, 3045, 3051, 3055, 3056, 3057, 3058, 3060, 3062, 3063, 3068, 3069, 3081, 3086, 3090, 3096, 3106, 3107, 3110, 3111, 3113, 3114, 3118, 3122, 3126, 3127, 3128, 3138, 3143, 3163, 3165, 3168, 3174, 3185, 3191, 3201, 3202, 3203, 3204, 3205, 3206, 3209, 3238, 3248, 3251, 3252, 3253, 3255, 3256, 3257, 3260, 3261, 3264, 3266, 3267, 3268, 3285, 3290, 3291, 3296, 3299, 3300, 3302, 3303, 3304, 3305, 3313, 3314, 3315, 3316, 3317, 3319, 3321, 3325, 3327, 3329, 3330, 3332, 3333, 3334, 3336, 3347, 3348, 3352, 3354, 3355, 3356, 3358, 3359, 3361, 3362, 3365, 3368, 3371, 3373, 3376, 3387, 3389, 3390, 3393, 3400, 3401, 3402, 3404, 3408, 3430, 3431, 3436, 3442, 3448, 3454, 3460, 3461, 3462, 3465, 3467, 3470, 3471, 3475, 3478, 3480, 3481, 3482, 3483, 3488, 3490, 3492, 3495, 3517, 3531, 3532, 3533, 3539, 3540, 3541, 3544, 3545, 3548, 3551, 3552, 3608, 3609, 3610, 3611, 3619, 3621, 3622, 3623, 3638, 3639, 3641, 3642, 3656, 3660, 3662, 3663, 3668, 3669, 3670, 3673, 3675, 3677, 3681, 3685, 3686, 3691, 3698, 3703, 3707, 3711, 3713, 3716, 3718, 3719, 3723, 3724, 3726, 3728, 3731, 3733, 3735, 3739, 3740, 3743, 3744, 3745, 3749, 3755, 3758, 3759, 3762, 3768, 3784, 3786, 3795, 3796, 3797, 3800, 3802, 3804, 3815, 3820, 3823, 3831, 3835, 3836, 3839, 3840, 3842, 3843, 3855, 3856, 3865, 3867, 3871, 3874, 3883, 3885, 3887, 3888, 3890, 3899, 3910, 3913, 3935, 3940, 3941, 3957, 3960, 3961, 3971, 3972, 3974, 3976, 3977, 3978, 3980, 3981, 3982, 3984, 3998, 3999, 4001, 4002, 4005, 4019, 4025, 4026, 4027, 4032, 4034, 4043, 4049, 4050, 4051, 4052, 4061, 4071, 4087, 4103, 4104, 4124, 4125, 4130, 4146, 4158, 4160, 4162, 4163, 4164, 4165, 4166, 4168, 4169, 4171, 4172, 4173, 4174, 4175, 4176, 4180, 4181, 4183, 4186, 4188, 4189, 4191, 4197, 4199, 4200, 4201, 4204, 4205, 4207, 4210, 4212, 4221, 4235, 4241, 4254, 4258, 4259, 4262, 4263, 4268, 4269, 4272, 4275, 4280, 4281, 4285, 4288, 4289, 4290, 4307, 4309, 4310, 4324, 4331, 4352, 4353, 4354, 4355, 4356, 4357, 4358, 4375, 4379, 4382, 4387, 4389, 4391, 4402, 4404, 4405, 4406, 4407, 4408, 4409, 4443, 4447, 4449, 4451, 4452, 4453, 4455, 4456, 4484, 4485, 4486, 4496, 4497, 4506, 4507, 4509, 4513, 4531, 4534, 4553, 4555, 4557, 4561, 4562, 4563, 4564, 4571, 4572, 4574, 4575, 4576, 4578, 4581, 4582, 4585, 4588, 4589, 4592, 4593, 4594, 4603, 4611, 4612, 4613, 4616, 4617, 4622, 4627, 4628, 4629, 4630, 4635, 4636, 4639, 4641, 4644, 4645, 4647, 4648, 4650, 4655, 4656, 4658, 4659, 4660, 4662, 4663, 4664, 4667, 4669, 4670, 4675, 4677, 4680, 4686, 4687, 4688, 4691, 4698, 4706, 4708, 4709, 4711, 4712, 4714, 4715, 4717, 4719, 4720, 4722, 4724, 4725, 4726, 4727, 4731, 4732, 4737, 4742, 4744, 4746, 4747, 4753, 4754, 4760, 4772, 4774, 4782, 4786, 4790, 4792, 4796, 4798, 4805, 4806, 4807, 4811, 4812, 4813, 4814, 4820, 4822, 4825, 4829, 4832, 4836, 4839, 4842, 4843, 4845, 4848, 4852, 4860, 4865, 4868, 4869, 4873, 4897, 4898, 4902, 4908, 4909, 4922, 4926, 4943, 4947, 4949, 4952, 4953, 4954, 4958, 4961, 4967, 4969, 4977, 4982, 4983, 4984, 4986, 4987, 4988, 4989, 4990, 4991, 4992, 4995, 4996, 4997, 4998, 5001, 5002, 5003, 5005, 5009, 5010, 5012, 5015, 5019, 5020, 5021, 5025, 5026, 5028, 5029, 5031, 5033, 5034, 5038, 5039, 5049, 5050, 5051, 5052, 5054, 5055, 5057, 5073, 5074, 5082, 5084, 5085, 5086, 5088, 5090, 5093, 5101, 5112, 5119, 5122, 5123, 5126, 5127, 5128, 5129, 5130, 5131, 5132, 5133, 5134, 5147, 5151, 5153, 5154, 5155, 5157, 5158, 5166, 5172, 5180, 5181, 5188, 5189, 5193, 5194, 5196, 5197, 5215, 5218, 5220, 5221, 5224, 5225, 5226, 5227, 5228, 5249, 5264, 5267, 5270, 5272, 5273, 5274, 5280, 5282, 5283, 5286, 5288, 5289, 5290, 5292, 5315, 5324, 5326, 5327, 5333, 5339, 5341, 5342, 5343, 5344, 5345, 5347, 5374, 5375, 5378, 5387, 5398, 5411, 5434, 5435, 5439, 5441, 5446, 5466, 5470, 5471, 5493, 5499, 5507, 5509, 5510, 5525, 5538, 5539, 5565, 5569, 5570, 5574, 5575, 5580, 5583, 5585, 5588, 5590, 5596, 5597, 5599, 5601, 5613, 5622, 5624, 5625, 5627, 5630, 5631, 5638, 5640, 5646, 5648, 5649, 5652, 5653, 5654, 5656, 5658, 5661, 5662, 5664, 5667, 5669, 5670, 5682, 5684, 5698, 5703, 5706, 5709, 5713, 5714, 5716, 5717, 5725, 5726, 5730, 5746, 5748, 5753, 5754, 5756, 5763, 5764, 5767, 5773, 5774, 5790, 5793, 5796, 5797, 5798, 5808, 5810, 5812, 5813, 5816, 5823, 5824, 5825, 5828, 5829, 5831, 5833, 5861, 5862, 5868, 5869, 5876, 5878, 5879, 5880, 5888, 5894, 5897, 5899, 5901, 5902, 5923, 5935, 5938, 5940, 5942, 5952, 5955, 5957, 5959, 5960, 5990, 5992, 5994, 5999, 6006, 6007, 6018, 6027, 6030, 6036, 6038, 6039, 6045, 6051, 6053, 6056, 6066, 6070, 6072, 6085, 6086, 6088, 6089, 6093, 6108, 6112, 6115, 6119, 6120, 6124, 6126, 6135, 6161, 6163, 6164, 6165, 6166, 6170, 6181, 6188, 6189, 6190, 6204, 6210, 6220, 6224, 6227, 6230, 6231, 6252, 6254, 6262, 6267, 6268, 6269, 6271, 6272, 6273, 6285, 6294, 6301, 6302, 6303, 6304, 6348, 6349, 6350, 6352, 6380, 6385, 6393, 6397, 6399, 6421, 6423, 6432, 6434, 6455, 6457, 6459, 6460, 6465, 6480, 6488, 6497, 6499, 6500, 6501, 6504, 6542, 6566, 6569, 6572, 6577, 6584, 6585, 6586, 6587, 6588, 6589, 6604, 6641, 6644, 6649, 6650, 6656, 6666, 6667, 6669, 6671, 6679, 6686, 6690, 6691, 6705, 6708, 6710, 6712, 6732, 6751, 6752, 6753, 6754, 6755, 6757, 6759, 6764, 6767, 6770, 6773, 6774, 6777, 6781, 6785, 6797, 6818, 6826, 6829, 6830, 6832, 6833, 6843, 6844, 6846, 6852, 6853, 6886, 6896, 6897, 6898, 6899, 6916, 6918, 6922, 6925, 6929, 6930, 6933, 6953, 6964, 6983, 6984, 7003, 7004, 7006, 7009, 7015, 7020, 7023, 7032, 7049, 7050, 7059, 7062, 7064, 7083, 7085, 7108, 7111, 7112, 7114, 7115, 7118, 7122, 7128, 7129, 7130, 7131, 7137, 7138, 7140, 7142, 7147, 7166, 7170, 7180, 7181, 7203, 7204, 7205, 7206, 7231, 7234, 7237, 7240, 7243, 7250, 7266, 7273, 7276, 7299, 7303, 7326, 7328, 7329, 7331, 7332, 7333, 7334, 7335, 7337, 7339, 7341, 7346, 7347, 7349, 7352, 7355, 7369, 7372, 7377, 7385, 7388, 7396, 7401, 7405, 7406, 7412, 7414, 7415, 7418, 7432, 7435, 7441, 7446, 7449, 7452, 7470, 7476, 7477, 7478, 7505, 7507, 7508, 7531, 7536, 7541, 7544, 7572, 7578, 7593, 7614, 7625, 7626, 7628, 7629, 7630, 7631, 7633, 7636, 7637, 7641, 7642, 7643, 7644, 7646, 7655, 7656, 7658, 7659, 7663, 7664, 7665, 7669, 7671, 7672, 7675, 7685, 7696, 7700, 7701, 7704, 7705, 7709, 7718, 7723, 7724, 7742, 7743, 7744, 7760, 7772, 7773, 7774, 7775, 7776, 7777, 7783, 7785, 7788, 7801, 7802, 7826, 7830, 7832, 7833, 7834, 7836, 7874, 7897, 7898, 7907, 7922, 7924, 7941, 7956, 7959, 7961, 7967, 7970, 7973, 7975, 7976, 7989, 8000, 8001, 8002, 8003, 8004, 8005, 8006, 8012, 8013, 8027, 8047, 8048, 8049, 8053, 8054, 8085, 8092, 8103, 8105, 8111, 8114, 8117, 8126, 8147, 8161, 8162, 8169, 8174, 8189, 8194, 8207, 8210, 8230, 8236, 8241, 8255, 8264, 8284, 8309, 8310, 8311, 8323, 8325, 8345, 8346, 8348, 8349, 8353, 8357, 8358, 8366, 8368, 8369, 8375, 8377, 8379, 8380, 8382, 8390, 8391, 8392, 8394, 8398, 8401, 8403, 8404, 8408, 8409, 8411, 8412, 8413, 8415, 8421, 8425, 8427, 8434, 8436, 8437, 8438, 8439, 8441, 8442, 8444, 8448, 8454, 8457, 8458, 8463, 8464, 8474, 8479, 8483, 8485, 8489, 8491, 8497, 8498, 8500, 8503, 8506, 8508, 8511, 8515, 8518, 8520, 8521, 8522, 8531, 8533, 8535, 8536, 8542, 8546, 8547, 8548, 8549, 8552, 8555, 8558, 8561, 8563, 8569, 8572, 8579, 8589, 8592, 8598, 8601, 8603, 8604, 8609, 8611, 8613, 8614, 8616, 8617, 8620, 8623, 8625, 8627, 8631, 8636, 8639, 8649, 8652, 8658, 8659, 8663, 8665, 8674, 8675, 8676, 8680, 8681, 8682, 8684, 8685, 8697, 8703, 8706, 8714, 8715, 8717, 8718, 8726, 8739, 8740, 8748, 8756, 8762, 8769, 8780, 8791, 8800, 8812, 8813, 8814, 8815, 8818, 8819, 8825, 8827, 8840, 8841, 8845, 8846, 8860, 8861, 8862, 8863, 8867, 8868, 8869, 8870, 8871, 8873, 8878, 8880, 8885, 8888, 8893, 8894, 8895, 8900, 8903, 8906, 8911, 8917, 8923, 8925, 8926, 8927, 8928, 8936, 8938, 8942, 8945, 8953, 8954, 8956, 8957, 8959, 8966, 8978, 8979, 8981, 8988, 8991, 8996, 8997, 9006, 9007, 9008, 9011, 9013, 9025, 9033, 9035, 9039, 9042, 9046, 9053, 9058, 9060, 9064, 9070, 9071, 9073, 9074, 9077, 9078, 9085, 9087, 9089, 9090, 9091, 9100, 9103, 9104, 9105, 9106, 9116, 9117, 9122, 9125, 9127, 9135, 9145, 9147, 9158, 9171, 9175, 9177, 9179, 9183, 9184, 9197, 9200, 9205, 9209, 9216, 9217, 9221, 9230, 9246, 9259, 9263, 9268, 9270, 9271, 9276, 9277, 9278, 9282, 9283, 9284, 9285, 9288, 9290, 9292, 9293, 9294, 9296, 9299, 9311, 9312, 9332, 9335, 9341, 9342, 9366, 9369, 9375, 9377, 9385, 9405, 9407, 9409, 9410, 9411, 9414, 9415, 9422, 9426, 9428, 9433, 9437, 9443, 9457, 9472, 9473, 9487, 9500, 9502, 9503, 9505, 9506, 9507, 9508, 9510, 9511, 9518, 9519, 9527, 9531, 9534, 9537, 9539, 9544, 9547, 9551, 9553, 9556, 9564, 9572, 9574, 9579, 9581, 9584, 9585, 9586, 9590, 9591, 9593, 9595, 9598, 9600, 9601, 9602, 9604, 9605, 9606, 9610, 9614, 9617, 9618, 9620, 9625, 9629, 9636, 9637, 9642, 9647, 9648, 9659, 9663, 9666, 9668, 9669, 9673, 9674, 9680, 9684, 9685, 9686, 9687, 9690, 9693, 9694, 9695, 9697, 9698, 9699, 9701, 9703, 9709, 9713, 9716, 9717, 9719, 9730, 9733, 9736, 9738, 9741, 9750, 9754, 9757, 9759, 9766, 9768, 9778, 9780, 9789, 9796, 9797, 9800, 9807, 9808, 9809, 9810, 9811, 9816, 9817, 9828, 9829, 9830, 9834, 9841, 9844, 9847, 9852, 9853, 9860, 9861, 9862, 9863, 9866, 9872, 9874, 9876, 9877, 9878, 9879, 9880, 9881, 9883, 9884, 9887, 9888, 9889, 9890, 9891, 9892, 9895, 9896, 9897, 9900, 9901, 9903, 9905, 9909, 9911, 9912, 9919, 9920, 9924, 9928, 9930, 9939, 9942, 9947, 9950, 9953, 9955, 9959, 9967, 9973, 9979, 9981, 9982, 9985, 9986, 9987, 9992, 9993, 9995, 9996, 9998, 9999, 10001, 10005, 10010, 10012, 10018, 10019, 10021, 10023, 10026, 10028, 10030, 10035, 10041, 10044, 10046, 10063, 10064, 10069, 10073, 10074, 10077, 10083, 10088, 10091, 10092, 10093, 10095, 10097, 10098, 10100, 10104, 10105, 10107, 10108, 10114, 10119, 10122, 10123, 10126, 10133, 10134, 10147, 10151, 10155, 10156, 10162, 10168, 10171, 10172, 10182, 10183, 10186, 10191, 10193, 10197, 10199, 10200, 10201, 10204, 10207, 10210, 10212, 10216, 10217, 10220, 10224, 10225, 10226, 10229, 10235, 10239, 10240, 10241, 10245, 10246, 10247, 10248, 10250, 10251, 10254, 10255, 10259, 10260, 10262, 10263, 10265, 10266, 10269, 10271, 10272, 10273, 10277, 10278, 10279, 10280, 10281, 10282, 10283, 10300, 10301, 10302, 10305, 10307, 10313, 10315, 10320, 10321, 10329, 10331, 10333, 10334, 10335, 10336, 10337, 10344, 10345, 10348, 10350, 10361, 10363, 10364, 10365, 10372, 10376, 10377, 10384, 10397, 10401, 10404, 10405, 10407, 10410, 10412, 10413, 10415, 10419, 10422, 10424, 10427, 10428, 10433, 10436, 10441, 10443, 10444, 10447, 10450, 10453, 10455, 10456, 10457, 10458, 10461, 10464, 10470, 10472, 10473, 10475, 10479, 10482, 10487, 10489, 10491, 10492, 10493, 10495, 10496, 10497, 10500, 10501, 10502, 10505, 10506, 10507, 10515, 10516, 10519, 10526, 10527, 10528, 10529, 10536, 10540, 10541, 10542, 10543, 10544, 10545, 10546, 10547, 10548, 10551, 10553, 10554, 10556, 10557, 10558, 10559, 10560, 10563, 10564, 10565, 10567, 10568, 10574, 10580, 10589, 10592, 10594, 10595, 10604, 10606, 10607, 10608, 10611, 10618, 10619, 10621, 10622, 10626, 10628, 10629, 10631, 10635, 10638, 10640, 10641, 10642, 10643, 10645, 10646, 10648, 10649, 10652, 10658, 10662, 10665, 10668, 10669, 10671, 10673, 10676, 10677, 10679, 10680, 10685, 10686, 10689, 10697, 10699, 10702, 10705, 10710, 10711, 10713, 10715, 10716, 10717, 10719, 10721, 10722, 10725, 10726, 10729, 10734, 10736, 10737, 10741, 10743, 10747, 10748, 10749, 10750, 10751, 10752, 10753, 10756, 10757, 10758, 10760, 10763, 10766, 10767, 10769, 10771, 10773, 10774, 10782, 10784, 10786, 10789, 10791, 10792, 10794, 10796, 10798, 10810, 10821, 10825, 10827, 10829, 10830, 10831, 10832, 10833, 10834, 10835, 10836, 10837, 10838, 10839, 10842, 10846, 10848, 10852, 10861, 10862, 10865, 10871, 10873, 10877, 10878, 10879, 10880, 10881, 10892, 10901, 10905, 10906, 10916, 10962, 10963, 10967, 10970, 11032, 11033, 11034, 11106, 11107, 11111, 11130, 11133, 11139, 11145, 11155, 11176, 11177, 11199, 11201, 11217, 11220, 11224, 11237, 11240, 11242, 11256, 11261, 11265, 11277, 11291, 11333, 11336, 11342, 11344, 11348, 11397, 11398, 11400, 11405, 11414, 11420, 11429, 11462, 11465, 11467, 11498, 11501, 11509, 11532, 11544, 11548, 11594, 11612, 11674, 11680, 11681, 11687, 11704, 11705, 11707, 11709, 11711, 11712, 11716, 11718, 11719, 11739, 11743, 11776, 11793, 11797, 11805, 11806, 11813, 11830, 11831, 11835, 11843, 11851, 11853, 11867, 11883, 11914, 11931, 11942, 11949, 11962, 11965, 11968, 11971, 11977, 11979, 11993, 11994, 11997, 12002, 12005, 12006, 12015, 12020, 12024, 12025, 12027, 12036, 12037, 12046, 12050, 12059, 12062, 12063, 12077, 12083, 12085, 12086, 12087, 12090, 12095, 12111, 12115, 12117, 12118, 12121, 12130, 12133, 12151, 12171, 12175, 12177, 12182, 12183, 12185, 12188, 12192, 12217, 12237, 12241, 12243, 12251, 12255, 12260, 12266, 12301, 12302, 12305, 12312, 12314, 12323, 12328, 12368, 12501, 12504, 12505, 12518, 12538, 12539, 12542, 12550, 12551, 12566, 12567, 12570, 12573, 12579, 12587, 12588, 12590, 12591, 12594, 12602, 12603, 12604, 12610, 12617, 12630, 12631, 12634, 12635, 12636, 12646, 12650, 12657, 12658, 12659, 12669, 12672, 12682, 12686, 12693, 12709, 12711, 12712, 12713, 12715, 12716, 12719, 12726, 12736, 12741, 12747, 12761, 12762, 12767, 12772, 12779, 12781, 12800, 12802, 12810, 12823, 12832, 12837, 12840, 12846, 12853, 12855, 12867, 12868, 12871, 12872, 12876, 12878, 12884, 12885, 12891, 12935, 12943, 12944, 12955, 12958, 12961, 12962, 12966, 12970, 12974, 12977, 13104, 13112, 13117, 13118, 13121, 13133, 13135, 13143, 13146, 13154, 13163, 13168, 13170, 13177, 13181, 13185, 13188, 13202, 13216, 13220, 13305, 13314, 13319, 13320, 13332, 13333, 13341, 13342, 13356, 13361, 13370, 13372, 13373, 13376, 13383, 13384, 13388, 13396, 13412, 13416, 13420, 13426, 13427, 13428, 13434, 13439, 13442, 13445, 13446, 13450, 13462, 13474, 13475, 13491, 13494, 13504, 13514, 13527, 13545, 13551, 13552, 13553, 13557, 13559, 13562, 13563, 13573, 13574, 13576, 13580, 13582, 13586, 13593, 13604, 13606, 13608, 13610, 13612, 13637, 14201, 14202, 14213, 14217, 14224, 14232, 14247, 14249, 14265, 14270, 14284, 14288, 14289, 14308, 14313, 14314, 14316, 14323, 14324, 14330, 14332, 14359, 14363, 14372, 14376, 14379, 14380, 14387, 14395, 14397, 14402, 14412, 14413, 14414, 14416, 14418, 14452, 14453, 14466, 14470, 14479, 14487, 14498, 14502, 14504, 14506, 14507, 14510, 14511, 14515, 14523, 14526, 14527, 14533, 14538, 14539, 14552, 14568, 14580, 14587, 14593, 14594, 14599, 14606, 14608, 14614, 14622, 14629, 14637, 14642, 14643, 14649, 14659, 14670, 14673, 14675, 14677, 14687, 14697, 14703, 14706, 14716, 14717, 14720, 14737, 14750, 14751, 14760, 14762, 14764, 14771, 14773, 14778, 14779, 14788, 14801, 14804, 14805, 14818, 14823, 14825, 14848, 14856, 14860, 14862, 14866, 14868, 14870, 14871, 14877, 14879, 14881, 14885, 14896, 14917, 14922, 14923, 14952, 14953, 15501, 15515, 15521, 15522, 15524, 15526, 15534, 15535, 15544, 15571, 15583, 15588, 15589, 15593, 15594, 15600, 15601, 15610, 15614, 15621, 15622, 15624, 15631, 15633, 15650, 15657, 15662, 15664, 15676, 15682, 15684, 15705, 15707, 15719, 15724, 15726, 15731, 15732, 15739, 15744, 15745, 15746, 15750, 15753, 15757, 15759, 15762, 15764, 15767, 15768, 15769, 15774, 15779, 15792, 15793, 15807, 15808, 15809, 15812, 15817, 15828, 15843, 15855, 15857, 15859, 16415, 16417, 16424, 16431, 16432, 16434, 16446, 16448, 16453, 16459, 16460, 16461, 16462, 16467, 16468, 16469, 16470, 16477, 16488, 16491, 16492, 16497, 16502, 16506, 16507, 16508, 16510, 16512, 16513, 16520, 16526, 16541, 16550, 16551, 16556, 16564, 16570, 16580, 16582, 16583, 16588, 16604, 16625, 16642, 16644, 16652, 16673, 16677, 16678, 16685, 16695, 16697, 16708, 16715, 16728, 16730, 16732, 16733, 16744, 16747, 16748, 16754, 16756, 16757, 16766, 16770, 16787, 16792, 16793, 16797, 16829, 16830, 16846, 16856, 16879, 16886, 16895, 16919, 16923, 16932, 16941, 16942, 16945, 16950, 16955, 16958, 16962, 16975, 16997, 17000, 17606, 17634, 17635, 17638, 17643, 17648, 17658, 17660, 17687, 17689, 17692, 17693, 17709, 17723, 17725, 17728, 17732, 17733, 17736, 17737, 17741, 17742, 17746, 17749, 17751, 17763, 17765, 17766, 17772, 17779, 17782, 17785, 17787, 17797, 17800, 17813, 17814, 17818, 17824, 17828, 17833, 17843, 17850, 17854, 17858, 17859, 17860, 17874, 17875, 17876, 17883, 17885, 17892, 17895, 17897, 17903, 17907, 17912, 17916, 17918, 17920, 17924, 18002, 18004, 18006, 18007, 18010, 18014, 18015, 18016, 18028, 18034, 18035, 18049, 18050, 18055, 18065, 18066, 18069, 18075, 18080, 18090, 18093, 18094, 18097, 18099, 18102, 18111, 18112, 18114, 18124, 18126, 18127, 18128, 18158, 18170, 18181, 18184, 18199, 18203, 18216, 18223, 18234, 18249, 18251, 19906, 19908, 19913, 19921, 19925, 19927, 19928, 19947, 19949, 19950, 19952, 19959, 19962, 19965, 20016, 20019, 20024, 20025, 20037, 20041, 20042, 20048, 20053, 20068, 20071, 20080, 20082, 20085, 20087, 20088, 20090, 20091, 20095, 20096, 20099, 20100, 20103, 20108, 20113, 20121, 20127, 20133, 20139, 20153, 20154, 20159, 20161, 20164, 20165, 20175, 20179, 20180, 20182, 20193, 20194, 20200, 20202, 20205, 20209, 20212, 20215, 20219, 20226, 20230, 20235, 20248, 20258, 20259, 20263, 20265, 20282, 20297, 20300, 20311, 20316, 20324, 20329, 20330, 20348, 20364, 20376, 20390, 20401, 20410, 20426, 20440, 20441, 20460, 20461, 20474, 20477, 20478, 20500, 20507, 20508, 20513, 20520, 20567, 22007, 22009, 22014, 22021, 22022, 22024, 22041, 22050, 22065, 22083, 22084, 22096, 22111, 22113, 22114, 22130, 22144, 22146, 22155, 22158, 22160, 22165, 22175, 22176, 22177, 22186, 22191, 22192, 22195, 22196, 22203, 22211, 22223, 22224, 22226, 22229, 22236, 22246, 22260, 22265, 22271, 22275, 22277, 22283, 22306, 22309, 22312, 22319, 22325, 22330, 22334, 22337, 22343, 22352, 22354, 22356, 22361, 22366, 22367, 22372, 22374, 22375, 22380, 22397, 22410, 22415, 22418, 22419, 22428, 22434, 22436, 22444, 22452, 22458, 22459, 22464, 22478, 22480, 22494, 22495, 22504, 22506, 22509, 22516, 22527, 22531, 22533, 22539, 22540, 22542, 22543, 22544, 22546, 22552, 22560, 22564, 22581, 22588, 22596, 22597, 22606, 22609, 22612, 22616, 22623, 22626, 22627, 22629, 22634, 22641, 22655, 22657, 22658, 22663, 22664, 22668, 22671, 22672, 22674, 22686, 22691, 22702, 22707, 22743, 22747, 22749, 22752, 22755, 22758, 22759, 24201, 24203, 24206, 24212, 24213, 24214, 24216, 24226, 24229, 24231, 24233, 24243, 24300, 24303, 24307, 24311, 24313, 24314, 24316, 24324, 24327, 24329, 24338, 24400, 24403, 24405, 24406, 24407, 24410, 24412, 24420, 24423, 24424, 24426, 24427, 24428, 24431, 24446, 24448, 24473, 24486, 24489, 24496, 24498, 24520, 24558, 24622, 24624, 24625, 24626, 24627, 24638, 24647, 24652, 24654, 24657, 24675, 24685, 24686, 24713, 24714, 24715, 24716, 24718, 24719, 24723, 24725, 24730, 24733, 24756, 24758, 24759, 24761, 24766, 24769, 24782, 24783, 24790, 24795, 24800, 24812, 24822, 24836, 24842, 24856, 24864, 24873, 24874, 24876, 24878, 24879, 24881, 24885, 24889, 24892, 24894, 24897, 24898, 24903, 24906, 24907, 24908, 24913, 24916, 24930, 24931, 24935, 24954, 24959, 24963, 24966, 24971, 24976, 24977, 24978, 24979, 24985, 24986, 24987, 24988, 24989, 24995, 24997, 24999, 25002, 25005, 25009, 25012, 25017, 25027, 25028, 25034, 25038, 25041, 25042, 25043, 25047, 25050, 25055, 25058, 25059, 25061, 25062, 25065, 25071, 25073, 25082, 25084, 25085, 25087, 25089, 25094, 25098, 25101, 25107, 25121, 25127, 25129, 25130, 25132, 25136, 25142, 25155, 25160, 25162, 25166, 25171, 25172, 25175, 25180, 25182, 25185, 25189, 25194, 25197, 25198, 25199, 25201, 25206, 25215, 25218, 25220, 25223, 25241, 25247, 25250, 25256, 25258, 25265, 25269, 25279, 25280, 25282, 25290, 25303, 25304, 25316, 25319, 25324, 25329, 25330, 25331, 25334, 25347, 25349, 25352, 25356, 25359, 25362, 25366, 25367, 25368, 25369, 25371, 25373, 25382, 25385, 25387, 25389, 25390, 25393, 25396, 25406, 25418, 25436, 25442, 25447, 25450, 25452, 25458, 25462, 25464, 25465, 25471, 25474, 25475, 25477, 25478, 25480, 25481, 25485, 25487, 25501, 25507, 25509, 25510, 25526, 25540, 25541, 25543, 25544, 25550, 25557, 25561, 25571, 25577, 25580, 25589, 25591, 25592, 25599, 25611, 25615, 25624, 25640, 25642, 25644, 25647, 25654, 25655, 25669, 25671, 25674, 25675, 25681, 25697, 25698, 25700, 25703, 25709, 25712, 25716, 25717, 25722, 25724, 25725, 25727, 25733, 25737, 25749, 25751, 25767, 25770, 25779, 25783, 25785, 25795, 25797, 25798, 25799, 25803, 25804, 25814, 25825, 25831, 25832, 25838, 25842, 25843, 25850, 25851, 25853, 25855, 25882, 25885, 25895, 25899, 25901, 25914, 25921, 25922, 25926, 25927, 25929, 25941, 25950, 25954, 25958, 25960, 25961, 25969, 25970, 25976, 25984, 25988, 25991, 25995, 25999, 26001, 26006, 26015, 26017, 26019, 26025, 26028, 26033, 26035, 26044, 26045, 26053, 26068, 26075, 26083, 26084, 26092, 26094, 26108, 26110, 26115, 26117, 26129, 26130, 26134, 26150, 26152, 26155, 26157, 26158, 26164, 26168, 26171, 26172, 26202, 26219, 26228, 26239, 26244, 26245, 26254, 26269, 26271, 26272, 26274, 26292, 26293, 26297, 26303, 26304, 26305, 26311, 26335, 26341, 26345, 26352, 26362, 26365, 26368, 26378, 26379, 26383, 26384, 26391, 26392, 26403, 26416, 26424, 26434, 26436, 26454, 26460, 26465, 26466, 26476, 26482, 26501, 26516, 26524, 26532, 26538, 26543, 30002, 30003, 30006, 30013, 31002, 31005, 31010, 31012, 31019, 31021, 31030, 31036, 31037, 31047, 31052, 31064, 31065, 31068, 31074, 31078, 31080, 31081, 31083, 31085, 31087, 31106, 31121, 31128, 31135, 31153, 31161, 31190, 31193, 31194, 31201, 31221, 31253, 31258, 31266, 31274, 31276, 31278, 31293, 31295, 31302, 31306, 31327, 31332, 31337, 31340, 31342, 31343, 31373, 31378, 31379, 31387, 31388, 31392, 31394, 31425, 31432, 31445, 31495, 31499, 31504, 31506, 31509, 32100, 32140, 40017, 40023, 40024, 40025, 40027, 40029, 40033, 40038, 40043, 40047, 40054, 40064, 40065, 40080, 40099, 40101, 40114, 40126, 40155, 40171, 40176, 40177, 40184, 40189, 40194, 40202, 40208, 40242, 40252, 40253, 40256, 40257, 40278, 40282, 40284, 40285, 40288, 40289, 40296, 40298, 40301, 40310, 40316, 40319, 40320, 40329, 40331, 40340, 40343, 40351, 40354, 40370, 40371, 40383, 40393, 40399, 40403, 40422, 40424, 40459, 41009, 41109, 41117, 41124, 41135, 41264, 41357, 41421, 41439, 41510, 41658, 41702, 41773, 41780, 41791, 41794, 41814, 41899
    ]

# Crear un DataFrame para guardar los resultados
results = []

# Definir el modelo y parámetros
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name='gpt-4o', temperature=0, max_tokens=5)

# Define los prompts específicos con ejemplos
prompts = [
    ("PROHIBE_JORNADA", """Rol: Eres experto en análisis cualitativo.
Objetivo: Debes hacer una codificación deductiva de manuales de convivencia escolar de Chile considerando la categoría que se indica a continuación.
Input: Manual de convivencia escolar de un colegio de Chile en formato PDF.
Categoría:
[
    {
        "Nombre": "Prohíbe Uso Durante Horario Escolar",
        "Definición": "Normativas que prohíben de manera absoluta el uso de dispositivos móviles durante toda la jornada escolar. Esto incluye no permitir que los estudiantes lleven dispositivos móviles al colegio, y prohibir su uso en clases, recreos, almuerzos y cualquier otra actividad escolar.",
        "Nombre Variable": "PROHIBE_JORNADA",
        "Pregunta Guía": "¿Se prohíbe explícitamente llevar o utilizar dispositivos móviles o tecnología en el establecimiento educacional, incluyendo la prohibición de su uso en clases, recreo, almuerzo y en otras actividades escolares?",
        "Consideraciones": "Aplicable solo si la prohibición es total y no se mencionan excepciones en el documento. Esta es la restricción más estricta y abarca todas las situaciones posibles durante la jornada escolar. Es excluyente con las categorías 'PROHIBE_CLASES', 'RESTRINGE_USO' y 'SIN_REGULACIONES'"
    },
    {
        "Nombre": "Prohíbe Uso Durante Clases",
        "Definición": "Normativas que prohíben de manera absoluta el uso de dispositivos móviles exclusivamente durante las sesiones de clase. Los estudiantes pueden llevar los dispositivos al colegio y usarlos en otras actividades como recreos y almuerzos, pero no pueden usarlos en el aula durante el horario de clase.",
        "Nombre Variable": "PROHIBE_CLASES",
        "Pregunta Guía": "¿Se prohíbe explícitamente el uso de dispositivo móviles, celulares o tecnología solamente en clases o aula y no en el resto de la jornada, recreo o almuerzo?",
        "Consideraciones": "Aplicable solo si la prohibición es total dentro del aula y no permite excepciones durante las sesiones de clase. Esta categoría es menos estricta que 'Prohíbe Uso Durante Horario Escolar'. Es excluyente con las categorías 'PROHIBE_JORNADA', 'RESTRINGE_USO' y 'SIN_REGULACIONES'"
    },
    {
        "Nombre": "Restringe Uso Durante Clases",
        "Definición": "Normativas que restringen el uso de dispositivos móviles durante las clases pero permiten ciertas excepciones. Estas normativas no prohíben el uso de dispositivos móviles de manera absoluta ni durante toda la jornada escolar ni durante las sesiones de clase.",
        "Nombre Variable": "RESTRINGE_USO",
        "Pregunta Guía": "¿Existen excepciones específicas que autorizan el uso de dispositivos móviles y tecnología en clases?",
        "Consideraciones": "Aplicable si el documento permite el uso de dispositivos móviles bajo ciertas condiciones o excepciones dentro del aula y durante las clases. Esta categoría es más flexible y permite ciertos usos autorizados. Es excluyente con las categorías 'PROHIBE_CLASES', 'PROHIBE_JORNADA' y 'SIN_REGULACIONES'"
    },
    {
        "Nombre": "Sin Regulación",
        "Definición": "Indica que no se encontraron regulaciones o normas específicas relacionadas con el uso de dispositivos móviles y tecnología en el documento revisado. No hay restricciones ni limitaciones sobre el uso de estos dispositivos.",
        "Nombre Variable": "SIN_REGULACION",
        "Pregunta Guía": "¿No existen secciones que regulen el uso de dispositivos móviles o tecnología?",
        "Consideraciones": "Aplicable si el documento carece completamente de normas o regulaciones sobre el uso de dispositivos móviles, indicando una ausencia total de restricciones. Es excluyente con las categorías 'PROHIBE_CLASES', 'RESTRINGE_USO' y 'PROHIBE_CLASES'"
    }
]

Ejemplos Positivos:
[
    "No traer al Colegio materiales ajenos al quehacer educativo, objetos cortantes, armas blancas o de fuego, celulares, o cualquier otro sistema electrónico audible y elementos innecesarios al quehacer escolar.",
    "El estudiante no podrá portar celulares, parlantes, bazookas, mp4, notebook, netbook, cámara fotográficas o de video, tablets, juegos electrónicos, u otras herramientas tecnológicas que no sean necesarias para sus aprendizajes.",
    "El Celular será retirado a la entrada al Establecimiento y será devuelto al término de la Jornada Escolar.",
    "El uso del celular queda estrictamente prohibido al interior del establecimiento (aula, patio, comedor o cualquier dependencia del establecimiento).",
    "Está prohibido traer celular o cualquier aparato electrónico al establecimiento por parte de los estudiantes.",
    "Estará PROHIBIDO que los estudiantes asistan al colegio con celulares, tablets u otros dispositivos análogos."
]

Ejemplos Negativos:
[
    "Uso de celulares permitido durante recreos y almuerzos, pero no en clases.",
    "Uso de celulares en actividades académicas específicas con permiso del docente.",
    "Uso de dispositivos móviles permitido si no interrumpe las clases.",
    "Queda prohibido el uso de celulares u otros aparatos electrónicos personales que interrumpan el trabajo pedagógico en el aula.",
    "El uso de celulares estará permitido en casos excepcionales y debidamente justificados, como herramienta educativa específica y bajo la supervisión del docente responsable.",
    "Uso de aparatos electrónicos durante la clase sin autorización del profesor. Ej. celular, reproductor de música, Tablet.",
    "Los apoderados no deben llamar por celular a sus estudiantes durante las horas de clase; por lo tanto, procurarán hacerlo durante los periodos de recreos."
]

Tareas:
1. **Identificación de Acápites Relacionados**:
   - Utiliza funciones de búsqueda (por ejemplo, Ctrl + F) para identificar los principales acápites referentes al uso de celulares, dispositivos móviles y sus sanciones asociadas.
   - Prioriza los siguientes términos de búsqueda: celular(es), teléfonos móviles, tablet, smartphone, dispositivo(s), tecnológico(s)/tecnología, ciberacoso y ciberbullying (considerando que a veces el prefijo ciber- es escrito cyber-).

2. **Análisis y Codificación**:
   - Analiza cada acápite identificado y determina si la categoría "Prohíbe Uso Durante Horario Escolar" le corresponde, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de la categoría.

Output: 
Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, y responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.""", "PROHIBE_JORNADA"),

    ("PROHIBE_CLASES", """Rol: Eres experto en análisis cualitativo.
Objetivo: Debes hacer una codificación deductiva de manuales de convivencia escolar de Chile considerando la categoría que se indica a continuación.
Input: Manual de convivencia escolar de un colegio de Chile en formato PDF.
Categoría:
[
    {
        "Nombre": "Prohíbe Uso Durante Horario Escolar",
        "Definición": "Normativas que prohíben de manera absoluta el uso de dispositivos móviles durante toda la jornada escolar. Esto incluye no permitir que los estudiantes lleven dispositivos móviles al colegio, y prohibir su uso en clases, recreos, almuerzos y cualquier otra actividad escolar.",
        "Nombre Variable": "PROHIBE_JORNADA",
        "Pregunta Guía": "¿Se prohíbe explícitamente llevar o utilizar dispositivos móviles o tecnología en el establecimiento educacional, incluyendo la prohibición de su uso en clases, recreo, almuerzo y en otras actividades escolares?",
        "Consideraciones": "Aplicable solo si la prohibición es total y no se mencionan excepciones en el documento. Esta es la restricción más estricta y abarca todas las situaciones posibles durante la jornada escolar. Es excluyente con las categorías 'PROHIBE_CLASES', 'RESTRINGE_USO' y 'SIN_REGULACIONES'"
    },
    {
        "Nombre": "Prohíbe Uso Durante Clases",
        "Definición": "Normativas que prohíben de manera absoluta el uso de dispositivos móviles exclusivamente durante las sesiones de clase. Los estudiantes pueden llevar los dispositivos al colegio y usarlos en otras actividades como recreos y almuerzos, pero no pueden usarlos en el aula durante el horario de clase.",
        "Nombre Variable": "PROHIBE_CLASES",
        "Pregunta Guía": "¿Se prohíbe explícitamente el uso de dispositivo móviles, celulares o tecnología solamente en clases o aula y no en el resto de la jornada, recreo o almuerzo?",
        "Consideraciones": "Aplicable solo si la prohibición es total dentro del aula y no permite excepciones durante las sesiones de clase. Esta categoría es menos estricta que 'Prohíbe Uso Durante Horario Escolar'. Es excluyente con las categorías 'PROHIBE_JORNADA', 'RESTRINGE_USO' y 'SIN_REGULACIONES'"
    },
    {
        "Nombre": "Restringe Uso Durante Clases",
        "Definición": "Normativas que restringen el uso de dispositivos móviles durante las clases pero permiten ciertas excepciones. Estas normativas no prohíben el uso de dispositivos móviles de manera absoluta ni durante toda la jornada escolar ni durante las sesiones de clase.",
        "Nombre Variable": "RESTRINGE_USO",
        "Pregunta Guía": "¿Existen excepciones específicas que autorizan el uso de dispositivos móviles y tecnología en clases?",
        "Consideraciones": "Aplicable si el documento permite el uso de dispositivos móviles bajo ciertas condiciones o excepciones dentro del aula y durante las clases. Esta categoría es más flexible y permite ciertos usos autorizados. Es excluyente con las categorías 'PROHIBE_CLASES', 'PROHIBE_JORNADA' y 'SIN_REGULACIONES'"
    },
    {
        "Nombre": "Sin Regulación",
        "Definición": "Indica que no se encontraron regulaciones o normas específicas relacionadas con el uso de dispositivos móviles y tecnología en el documento revisado. No hay restricciones ni limitaciones sobre el uso de estos dispositivos.",
        "Nombre Variable": "SIN_REGULACION",
        "Pregunta Guía": "¿No existen secciones que regulen el uso de dispositivos móviles o tecnología?",
        "Consideraciones": "Aplicable si el documento carece completamente de normas o regulaciones sobre el uso de dispositivos móviles, indicando una ausencia total de restricciones. Es excluyente con las categorías 'PROHIBE_CLASES', 'RESTRINGE_USO' y 'PROHIBE_CLASES'"
    }
]
Ejemplos Positivos:
[
    "Está prohibido el uso de objetos que no correspondan y dificulten el desarrollo de la clase, como cámaras de video o fotográficas, celulares, pendrive y otros similares.",
    "Los teléfonos celulares deben permanecer apagados y guardados durante el horario de clases. Su uso está permitido únicamente en los recreos y momentos de transición entre clases.",
    "Está prohibido mantener encendidos o usar equipos personales de audio o de telefonía móvil en clases y en toda actividad escolar.",
    "Hablar o utilizar para mensajería, navegación, juego o en general, mantener encendido un teléfono celular propio o ajeno en la sala en horario de clases."
]
Ejemplos Negativos:
[
    "Está prohibido utilizar durante el desarrollo de la clase teléfonos celulares, mp3, mp4, máquinas fotográficas, filmadoras sin autorización del profesor que se encuentre en aula o por otro estamento de la escuela.",
    "Permiso para usar dispositivos electrónicos en actividades pedagógicas.",
    "Uso de celulares restringido durante clases, con excepciones autorizadas.",
    "Uso de celulares durante clases permitido en casos específicos autorizados por el docente."
    "Utilizar elementos distractores en clases como escuchar música, jugar con celular u otro implemento electrónico, juego o elemento no autorizado.",
    "Prohibido el uso del celular durante el transcurso de las clases a no ser que sea utilizado de manera pedagógica y el profesor lo autorice en su clase."
]
Tareas:
1. **Identificación de Acápites Relacionados**:
   - Utiliza funciones de búsqueda (por ejemplo, Ctrl + F) para identificar los principales acápites referentes al uso de celulares, dispositivos móviles y sus sanciones asociadas.
   - Prioriza los siguientes términos de búsqueda: celular(es), teléfonos móviles, tablet, smartphone, dispositivo(s), tecnológico(s)/tecnología, ciberacoso y ciberbullying (considerando que a veces el prefijo ciber- es escrito cyber-).

2. **Análisis y Codificación**:
   - Analiza cada acápite identificado y determina si la categoría "Prohíbe Uso Durante Clases" le corresponde, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de la categoría.
Output: Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, y responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.""", "PROHIBE_CLASES"),

    ("CONFISCACION", """Rol: Eres experto en análisis cualitativo.
Objetivo: Debes hacer una codificación deductiva de manuales de convivencia escolar de Chile considerando la categoría que se indica a continuación.
Input: Manual de convivencia escolar de un colegio de Chile en formato PDF.
Categoría:
[
    {
        "Nombre": "Establece Protocolos de Confiscación de Dispositivos",
        "Definición": "Esta categoría se refiere a los reglamentos que detallan los procedimientos a seguir cuando un dispositivo móvil es utilizado inapropiadamente o sin autorización. Incluye detalles sobre cómo y cuándo se confiscarán los dispositivos y las condiciones de su devolución.",
        "Nombre Variable": "CONFISCACION",
        "Pregunta Guía": "¿Existen normas que incluyen la confiscación de dispositivos móviles o tecnología y los procedimientos asociados?",
        "Consideraciones": "Aplicable en cualquier caso donde se mencionen reglas específicas para la confiscación de dispositivos móviles. Se incluye la entrega voluntaria o por solicitud del docente a cargo, al ingresar al colegio, al utilizarlo en clases, al interrumpir la clase"
    }
]
Ejemplos Positivos:
[
    "El mal uso del celular por parte del alumno(a) faculta al docente a retirarlo y posteriormente entregarlo a su apoderado personalmente. En el caso del uso pedagógico de Notebooks o Tablet dentro del establecimiento. El apoderado y el alumno(a) asumen la responsabilidad en el uso, daño o pérdida del artículo.",
    "El Celular será retirado a la entrada al Establecimiento y será devuelto al término de la Jornada Escolar.",
    "El docente solicita a alumno la entrega de artefacto tecnológico.",
    "Es importante mencionar que el alumno o alumna que porte este tipo de elementos sin autorización se procederá a su retiro y será entregado en Inspectoría General, desde allí se citará al apoderado correspondiente para que retire el artículo. En caso de reincidencia el aparato será retenido y devuelto al finalizar el año escolar.",
    "Al ser sorprendido usando estos equipos durante el desarrollo de actividades académicas, ellos serán retirados por el profesor respectivo quien consignará lo ocurrido en la hoja de vida del alumno y devolverá el equipo al estudiante al finalizar la jornada. En caso de repetirse la conducta, se informará al apoderado(a) y si esta conducta persiste se procederá a retirarlo hasta finalizar el año escolar siendo entregado al apoderado(a).",
    "La Escuela retirará estos objetos en caso de no cumplir con lo que se especifica en los puntos anteriores y a tomar las medidas formativas y disciplinarias que correspondan. Los objetos tecnológicos serán devueltos al apoderado al finalizar la jornada escolar.",
    "El establecimiento no se hará responsable de daños, pérdidas o sustracciones ocurridas fuera del horario establecido para la entrega y devolución de los dispositivos móviles."
]
Ejemplos Negativos:
[
    "Uso de dispositivos móviles permitido siempre que no interrumpa el desarrollo de actividades escolares.",
    "No se menciona ningún procedimiento específico para la confiscación de dispositivos móviles.",
    "Se autoriza el uso en actividades académicas y formativas de dispositivos electrónicos tales como: celulares, MP3, MP4, pendrive, notebooks y otros similares, cuando el docente a cargo lo haya autorizado para el desarrollo de alguna actividad pedagógica específica.",
    "Se prohibe utilizar equipos tecnológicos propios y del colegio de manera responsable y solo cuando está autorizado por el docente o el educador a cargo de la actividad."
    "Utilizar elementos distractores en clases como escuchar música, jugar con celular u otro implemento electrónico, juego o elemento no autorizado.",
    "Prohibido el uso del celular durante el transcurso de las clases a no ser que sea utilizado de manera pedagógica y el profesor lo autorice en su clase."
]
Tareas:
1. **Identificación de Acápites Relacionados**:
   - Utiliza funciones de búsqueda (por ejemplo, Ctrl + F) para identificar los principales acápites referentes al uso de celulares, dispositivos móviles, objetos tecnológicos, y sus sanciones asociadas.
   - Prioriza los siguientes términos de búsqueda: objetos, devolución, entrega, retirar, confiscar, requisar, celular(es), teléfonos móviles, tablet, smartphone, dispositivo(s), tecnológico(s)/tecnología, ciberacoso y ciberbullying (considerando que a veces el prefijo ciber- es escrito cyber-).

2. **Análisis y Codificación**:
   - Analiza cada acápite identificado y determina si la categoría "Establece Protocolos de Confiscación de Dispositivos" le corresponde, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de la categoría.
Output: Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, y responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.""", "CONFISCACION"),

    ("PROHIBE_FOTOS", """Rol: Eres experto en análisis cualitativo.
Objetivo: Debes hacer una codificación deductiva de manuales de convivencia escolar de Chile considerando la categoría que se indica a continuación.
Input: Manual de convivencia escolar de un colegio de Chile en formato PDF.
Categoría:
[
    {
        "Nombre": "Prohíbe Grabaciones y Fotos Dentro del Establecimiento Sin Autorización",
        "Definición": "Esta categoría abarca las prohibiciones de uso de tecnología y dispositivos móviles para capturar imágenes o grabar videos dentro de las instalaciones escolares sin el permiso explícito de la administración escolar o del docente, y establece sanciones para quienes falten a esta normativa.",
        "Nombre Variable": "PROHIBE_FOTOS",
        "Pregunta Guía": "¿Se prohíbe explícitamente fotografiar y grabar sin autorización?",
        "Consideraciones": "Solo se aplica cuando en el documento se mencionan restricciones y prohibiciones de grabaciones y fotografías"
    }
]
Ejemplos Positivos:
[
    "No está permitido tomar fotos o videos sin autorización respectiva en ningún área de la Escuela, con el fin de respetar la integridad y privacidad de cada miembro de la comunidad educativa y de evitar publicaciones que puedan ser privadas u ofensivas.",
    "Se prohibe fotografiar pruebas, guías evaluadas, libro de clases, bases de datos del colegio o cualquier otro documento sin autorización.",
    "Está prohibido tomar fotografías a compañeros o docentes para generar burlas o memes.",
    "Está prohibido sacar fotos, videos o capturas de pantalla sin autorización, que sean o no ofensivas para sus compañeras, padres y apoderados, funcionarios y/o profesores de la Comunidad Educativa.",
    "Fotografiar, filmar en clases o grabar conversaciones con docentes, asistentes de la educación o Equipo Directivo del Colegio o estudiantes, con cualquier medio electrónico, sin autorización o contra la voluntad del tercero.",
    "Uso inadecuado del teléfono celular, como tomar fotografías o videos sin la autorización respectiva.",
    "Subir fotografías, videos, imágenes u otros a la red informática que atente contra la dignidad de las personas o que perjudique la imagen de la comunidad educativa."
]
Ejemplos Negativos:
[
    "No hay menciones explícitas sobre la prohibición de fotos y grabaciones.",
    "El establecimiento no se responsabiliza por la pérdida o destrozo de estos objetos o cualquier otro de valor que porten los estudiantes dentro del colegio.",
    "No se prohíbe explícitamente el uso de dispositivos para grabaciones y fotos sin autorización.",
    "Se prohibe utilizar equipos tecnológicos propios y del colegio de manera responsable y solo cuando está autorizado por el docente o el educador a cargo de la actividad."
    "Utilizar elementos distractores en clases como escuchar música, jugar con celular u otro implemento electrónico, juego o elemento no autorizado.",
    "Prohibido el uso del celular durante el transcurso de las clases a no ser que sea utilizado de manera pedagógica y el profesor lo autorice en su clase."
]
Tareas:
1. **Identificación de Acápites Relacionados**:
   - Utiliza funciones de búsqueda (por ejemplo, Ctrl + F) para identificar los principales acápites referentes al uso y toma de fotografías, grabaciones y sus sanciones asociadas.
   - Prioriza los siguientes términos de búsqueda: foto, videos, grabar, fotografiar, dispositivo(s), tecnológico(s)/tecnología, ciberacoso y ciberbullying (considerando que a veces el prefijo ciber- es escrito cyber-).

2. **Análisis y Codificación**:
   - Analiza cada acápite identificado y determina si la categoría "Prohíbe Grabaciones y Fotos Dentro del Establecimiento Sin Autorización" le corresponde, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de la categoría.
Output: Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, y responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.""", "PROHIBE_FOTOS"),

    ("RESTRINGE_USO", """Rol: Eres experto en análisis cualitativo.
Objetivo: Debes hacer una codificación deductiva de manuales de convivencia escolar de Chile considerando la categoría que se indica a continuación.
Input: Manual de convivencia escolar de un colegio de Chile en formato PDF.
Categoría:
[
    {
        "Nombre": "Prohíbe Uso Durante Horario Escolar",
        "Definición": "Normativas que prohíben de manera absoluta el uso de dispositivos móviles durante toda la jornada escolar. Esto incluye no permitir que los estudiantes lleven dispositivos móviles al colegio, y prohibir su uso en clases, recreos, almuerzos y cualquier otra actividad escolar.",
        "Nombre Variable": "PROHIBE_JORNADA",
        "Pregunta Guía": "¿Se prohíbe explícitamente llevar o utilizar dispositivos móviles o tecnología en el establecimiento educacional, incluyendo la prohibición de su uso en clases, recreo, almuerzo y en otras actividades escolares?",
        "Consideraciones": "Aplicable solo si la prohibición es total y no se mencionan excepciones en el documento. Esta es la restricción más estricta y abarca todas las situaciones posibles durante la jornada escolar. Es excluyente con las categorías 'PROHIBE_CLASES', 'RESTRINGE_USO' y 'SIN_REGULACIONES'"
    },
    {
        "Nombre": "Prohíbe Uso Durante Clases",
        "Definición": "Normativas que prohíben de manera absoluta el uso de dispositivos móviles exclusivamente durante las sesiones de clase. Los estudiantes pueden llevar los dispositivos al colegio y usarlos en otras actividades como recreos y almuerzos, pero no pueden usarlos en el aula durante el horario de clase.",
        "Nombre Variable": "PROHIBE_CLASES",
        "Pregunta Guía": "¿Se prohíbe explícitamente el uso de dispositivo móviles, celulares o tecnología solamente en clases o aula y no en el resto de la jornada, recreo o almuerzo?",
        "Consideraciones": "Aplicable solo si la prohibición es total dentro del aula y no permite excepciones durante las sesiones de clase. Esta categoría es menos estricta que 'Prohíbe Uso Durante Horario Escolar'. Es excluyente con las categorías 'PROHIBE_JORNADA', 'RESTRINGE_USO' y 'SIN_REGULACIONES'"
    },
    {
        "Nombre": "Restringe Uso Durante Clases",
        "Definición": "Normativas que restringen el uso de dispositivos móviles durante las clases pero permiten ciertas excepciones. Estas normativas no prohíben el uso de dispositivos móviles de manera absoluta ni durante toda la jornada escolar ni durante las sesiones de clase.",
        "Nombre Variable": "RESTRINGE_USO",
        "Pregunta Guía": "¿Existen excepciones específicas que autorizan el uso de dispositivos móviles y tecnología en clases?",
        "Consideraciones": "Aplicable si el documento permite el uso de dispositivos móviles bajo ciertas condiciones o excepciones dentro del aula y durante las clases. Esta categoría es más flexible y permite ciertos usos autorizados. Es excluyente con las categorías 'PROHIBE_CLASES', 'PROHIBE_JORNADA' y 'SIN_REGULACIONES'"
    },
    {
        "Nombre": "Sin Regulación",
        "Definición": "Indica que no se encontraron regulaciones o normas específicas relacionadas con el uso de dispositivos móviles y tecnología en el documento revisado. No hay restricciones ni limitaciones sobre el uso de estos dispositivos.",
        "Nombre Variable": "SIN_REGULACION",
        "Pregunta Guía": "¿No existen secciones que regulen el uso de dispositivos móviles o tecnología?",
        "Consideraciones": "Aplicable si el documento carece completamente de normas o regulaciones sobre el uso de dispositivos móviles, indicando una ausencia total de restricciones. Es excluyente con las categorías 'PROHIBE_CLASES', 'RESTRINGE_USO' y 'PROHIBE_CLASES'"
    }
]
Ejemplos Positivos:
[
    "Para promover un clima escolar favorable al aprendizaje, no está permitido durante la hora de clases la utilización de aparatos electrónicos tales como; teléfonos celulares, cámaras fotográficas, aparatos reproductores de música, computadores, tablets, entre otros. El alumno podrá hacer uso de los objetos mencionados sólo con autorización del profesor.",
    "En relación al uso del celular por parte del estudiante, el establecimiento comprende que bajo ciertas circunstancias el estudiante podría hacer uso de él, siempre y cuando sea solicitado y firmado una carta de compromiso de buen uso por parte del apoderado, respetando los principios de igualdad, dignidad, inclusión y no discriminación, y que, si se ha actuado no conforme a estos derechos, quedará prohibido ingresar dicho elemento al establecimiento.",
    "Se autoriza el uso en actividades académicas y formativas de dispositivos electrónicos tales como: celulares, MP3, MP4, pendrive, notebooks y otros similares, cuando el docente a cargo lo haya autorizado para el desarrollo de alguna actividad pedagógica específica.",
    "Está prohibido utilizar celulares, equipos electrónicos personales, durante las horas de clases, sin la autorización del profesor.",
    "Se prohibe utilizar equipos tecnológicos propios y del colegio de manera responsable y solo cuando está autorizado por el docente o el educador a cargo de la actividad."
    "Utilizar elementos distractores en clases como escuchar música, jugar con celular u otro implemento electrónico, juego o elemento no autorizado.",
    "Prohibido el uso del celular durante el transcurso de las clases a no ser que sea utilizado de manera pedagógica y el profesor lo autorice en su clase."
]
Ejemplos Negativos:
[
    "Uso de celulares prohibido en todo momento sin excepciones.",
    "No se permiten dispositivos móviles dentro del establecimiento en ninguna circunstancia.",
    "Uso de tecnología móvil completamente prohibido durante la jornada escolar.",
    "Está prohibido el uso de objetos que no correspondan y dificulten el desarrollo de la clase, como cámaras de video o fotográficas, celulares, pendrive y otros similares.",
    "Hablar o utilizar para mensajería, navegación, juego o en general, mantener encendido un teléfono celular propio o ajeno en la sala en horario de clases."
]
Tareas:
1. **Identificación de Acápites Relacionados**:
   - Utiliza funciones de búsqueda (por ejemplo, Ctrl + F) para identificar los principales acápites referentes al uso de celulares, dispositivos móviles y sus sanciones asociadas.
   - Prioriza los siguientes términos de búsqueda: celular(es), teléfonos móviles, tablet, smartphone, dispositivo(s), tecnológico(s)/tecnología, ciberacoso y ciberbullying (considerando que a veces el prefijo ciber- es escrito cyber-).

2. **Análisis y Codificación**:
   - Analiza cada acápite identificado y determina si la categoría "Restringe Uso Durante Clases" le corresponde, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de la categoría.
Output: Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, y responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.""", "RESTRINGE_USO"),
    
    ("USO_INAPR", """Rol: Eres experto en análisis cualitativo.
Objetivo: Debes hacer una codificación deductiva de manuales de convivencia escolar de Chile considerando la categoría que se indica a continuación.
Input: Manual de convivencia escolar de un colegio de Chile en formato PDF.
Categoría:
[
    {
        "Nombre": "Regula Uso Inapropiado",
        "Definición": "Esta categoría incluye normativas que abordan el uso dispositivos móviles o tecnología de manera indebida, como el acceso a contenido inapropiado, el ciberacoso, o el uso de dispositivos para actividades disruptivas o no éticas.",
        "Nombre Variable": "USO_INAPR",
        "Pregunta Guía": "¿Se regula el tipo de uso inapropiado que los alumnos pueden darle a sus dispositivos móviles y tecnología?",
        "Consideraciones": "se aplica cuando en el documento existen reglamentosa (no solo definiciones), frente al ciberacoso, cyberbulling, distribución de material pornográfico, uso inapropiado de internet, entre otros. Se centra en el uso inapropiado y se excluye el uso apropiado"
    }
]
Ejemplos Positivos:
[
    "Se prohibe amenazar, atacar, injuriar o desprestigiar a un alumno o a cualquier otro integrante de la comunidad educativa a través de redes sociales, mensajes de texto, correos electrónicos, foros, servidores que almacenan videos o fotografías, sitios webs, teléfonos o cualquier otro medio tecnológico, virtual o electrónico, como también de manera verbal.",
    "Se prohibe exhibir, transmitir y/o difundir por medios cibernéticos cualquier conducta de maltrato escolar.",
    "Se prohibe hacer uso de Red Internet o de medios u objetos tecnológicos para: afectar la privacidad, la honra, ofender, amenazar, injuriar, calumniar, desprestigiar a cualquier integrante de la Comunidad Escolar, provocando daño psicológico al, o los afectados. (Ley de la Violencia Escolar 2.536)",
    "Se prohibe ejercer bullying, cyberbullying, acoso permanente a una persona.",
    "Se prohibe la utilización de medios cibernéticos o audiovisuales, para menoscabar la dignidad y honra de los/as estudiantes, funcionarios/as.",
    "Como comunidad educativa, se entenderá por maltrato escolar, cualquier acción intencional, ya sea física o psicológica, realizada en forma escrita, verbal o a través de medios tecnológicos o cibernéticos, en contra de cualquier integrante de la comunidad educativa, con independencia del lugar en que se cometa",
    "Amenazar, atacar, injuriar o desprestigiar a un alumno o a cualquier otro integrante de la comunidad educativa a través de chats, blogs, Facebook, mensajes de textos, correo electrónico, foros, servidores que almacenan videos o fotografías, sitios webs, teléfonos o cualquier otro medio tecnológico, virtual o electrónico"
]
Ejemplos Negativos:
[
    "Uso de dispositivos móviles no está regulado en relación con actividades inapropiadas.",
    "No se mencionan restricciones específicas sobre el uso inapropiado de tecnología.",
    "Se permite el uso de celulares en casos excepcionales y debidamente justificados, como herramienta educativa específica y bajo la supervisión del docente responsable.",
    "Hablar o utilizar para mensajería, navegación, juego o en general, mantener encendido un teléfono celular propio o ajeno en la sala en horario de clases."
]
Tareas:
1. **Identificación de Acápites Relacionados**:
   - Utiliza funciones de búsqueda (por ejemplo, Ctrl + F) para identificar los principales acápites referentes al uso inapropiado de tecnología, celulares, dispositivos móviles y sus sanciones asociadas.
   - Prioriza los siguientes términos de búsqueda: celular(es), teléfonos móviles, tablet, smartphone, dispositivo(s), tecnológico(s)/tecnología, ciberacoso y ciberbullying (considerando que a veces el prefijo ciber- es escrito cyber-).

2. **Análisis y Codificación**:
   - Analiza cada acápite identificado y determina si la categoría "Regula Uso Inapropiado" le corresponde, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de la categoría.
Output: Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, y responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.""", "USO_INAPR"),

    ("PROTOCOLO_USO", """Rol: Eres experto en análisis cualitativo.
Objetivo: Debes hacer una codificación deductiva de manuales de convivencia escolar de Chile considerando la categoría que se indica a continuación.
Input: Manual de convivencia escolar de un colegio de Chile en formato PDF.
Categoría:
[
    {
        "Nombre": "Establece Protocolos de Uso Durante Clases (Incluye Clases Online y Aula de Informática)",
        "Definición": "Esta categoría incluye normativas que definen orientaciones o protocolos para un buen uso de los dispositivos móviles y tecnología durante las clases, incluidas las clases virtuales y en aulas de informática, promoviendo un uso educativo y regulado.",
        "Nombre Variable": "PROTOCOLO_USO",
        "Pregunta Guía": "¿Existen protocolos que definen buenas prácticas u orientaciones para el buen uso de dispositivos móviles y tecnología durante clases y en otras instancias educativas?",
        "Consideraciones": "Aplicable solo si el documento incluye protocolos o reglamentos específicos sobre buenas prácticas y el uso adecuado de tecnología en entornos educativos. Se centra en el uso apropiado y se excluye el uso inapropiado. Se refiere a un conjunto de pasos a seguir para un buen uso de la tecnología en el aula o en las clases virtuales"
    }
]
Ejemplos Positivos:
[
    "NORMAS DE COMPORTAMIENTO DURANTE LA CLASE VIRTUAL.",
    "PROTOCOLO PROCEDIMIENTOS Y ORIENTACIONES PARA EL USO DE CELULARES Y OTROS DISPOSITIVOS MÓVILES EN EL COLEGIO.",
    "USO DE CELULAR Y APARATOS TECNOLÓGICOS.",
    "PROTOCOLO DE CONVIVENCIA DIGITAL."
]
Ejemplos Negativos:
[
    "No se mencionan protocolos específicos para el uso de dispositivos móviles.",
    "Uso de tecnología no regulado por ningún protocolo en particular.",
    "El establecimiento no se responsabiliza por la pérdida o destrozo de estos objetos o cualquier otro de valor que porten los estudiantes dentro del colegio.",
    "Se prohibe amenazar, atacar, injuriar o desprestigiar a un alumno o a cualquier otro integrante de la comunidad educativa a través de redes sociales, mensajes de texto, correos electrónicos, foros, servidores que almacenan videos o fotografías, sitios webs, teléfonos o cualquier otro medio tecnológico, virtual o electrónico, como también de manera verbal.",
    "Se prohibe exhibir, transmitir y/o difundir por medios cibernéticos cualquier conducta de maltrato escolar.",
    "Se prohibe hacer uso de Red Internet o de medios u objetos tecnológicos para: afectar la privacidad, la honra, ofender, amenazar, injuriar, calumniar, desprestigiar a cualquier integrante de la Comunidad Escolar, provocando daño psicológico al, o los afectados. (Ley de la Violencia Escolar 2.536)",
    "Para promover un clima escolar favorable al aprendizaje, no está permitido durante la hora de clases la utilización de aparatos electrónicos tales como; teléfonos celulares, cámaras fotográficas, aparatos reproductores de música, computadores, tablets, entre otros. El alumno podrá hacer uso de los objetos mencionados sólo con autorización del profesor.",
    "En relación al uso del celular por parte del estudiante, el establecimiento comprende que bajo ciertas circunstancias el estudiante podría hacer uso de él, siempre y cuando sea solicitado y firmado una carta de compromiso de buen uso por parte del apoderado, respetando los principios de igualdad, dignidad, inclusión y no discriminación, y que, si se ha actuado no conforme a estos derechos, quedará prohibido ingresar dicho elemento al establecimiento.",
    "Se autoriza el uso en actividades académicas y formativas de dispositivos electrónicos tales como: celulares, MP3, MP4, pendrive, notebooks y otros similares, cuando el docente a cargo lo haya autorizado para el desarrollo de alguna actividad pedagógica específica."
]
Tareas:
1. **Identificación de Acápites Relacionados**:
   - Utiliza funciones de búsqueda (por ejemplo, Ctrl + F) para identificar los principales acápites referentes a protocolos de uso de celulares, dispositivos móviles y sus sanciones asociadas.
   - Prioriza los siguientes términos de búsqueda: virtuales, protocolos, autorización, celular(es), teléfonos móviles, tablet, smartphone, dispositivo(s), tecnológico(s)/tecnología, ciberacoso y ciberbullying (considerando que a veces el prefijo ciber- es escrito cyber-).

2. **Análisis y Codificación**:
   - Analiza cada acápite identificado y determina si la categoría "Establece Protocolos de Uso Durante Clases (Incluye Clases Online y Aula de Informática)" le corresponde, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de la categoría.
Output: Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, y responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.""", "PROTOCOLO_USO"),

    ("PROHIBE_REDES", """Rol: Eres experto en análisis cualitativo.
Objetivo: Debes hacer una codificación deductiva de manuales de convivencia escolar de Chile considerando la categoría que se indica a continuación.
Input: Manual de convivencia escolar de un colegio de Chile en formato PDF.
Categoría:
[
    {
        "Nombre": "Prohíbe Contacto con el Establecimiento Vía Redes Sociales",
        "Definición": "Normativas que prohíben el uso de cuentas personales o redes sociales para comunicarse con el personal docente o entidades del colegio, requiriendo el uso de medios oficiales para dicha comunicación.",
        "Nombre Variable": "PROHIBE_REDES",
        "Pregunta Guía": "¿Se prohíbe que estudiantes o apoderados usen las redes sociales u otras plataformas no institucionales para contactarse con el personal docente del establecimiento?",
        "Consideraciones": "Aplicable solo si el documento menciona restricciones específicas sobre la comunicación entre estudiantes o apoderados y el personal del colegio mediante redes sociales no institucionales."
    }
]
Ejemplos Positivos:
[
    "No se permite el uso de cuentas personales en redes sociales para comunicarse con personal del colegio.",
    "Se prohíbe a Docentes, Directivos y Asistentes de la educación, la solicitud de “amistades” o “seguidores” a través de redes sociales con estudiantes o apoderados/as del Colegio. Toda vez que la relación pedagógica que prima en el establecimiento es necesariamente un vínculo formal y por tanto asimétrico, donde la figura del/la adulto(a) conlleva responsabilidades que no se extinguen por la sola existencia de un horario laboral y de forma de resguardar la privacidad de los espacios de intimidad de cada persona.",
    "En relación a las redes sociales, se prohíbe a todos los funcionarios del Colegio mantener algún tipo de conversación personal con alumnos del establecimiento por medio de estos canales virtuales (Facebook, Whatsapp, Skype, Instagram, Twitter, otros).",
    "Se debe considerar que, a modo preventivo y teniendo en cuenta el valor del espacio personal y la intimidad, los/as estudiantes no mantengan contacto a través de las redes sociales con las personas adultas que laboran en el establecimiento educativo. Sólo debería permitirse, si el profesor(a) jefe lo estima necesario, mantener un correo electrónico del curso para acoger preguntas de los alumno/as y/o Apoderados/as, así como para transmitir información importante respecto a la asignatura (MINEDUC, 27).",
    "Todo contacto por medio de redes virtuales entre los alumnos y el Colegio, incluyendo a los funcionarios, debe ser realizado a través de cuentas institucionales y no personales, por lo tanto, también queda prohibido al personal del Colegio que incluyan a los alumnos como contactos de sus redes sociales personales, salvo fines pedagógicos. El Colegio no responderá por dichos, actos, imágenes y/o situaciones relacionadas con redes personales entre sus funcionarios y familias y alumnos del Colegio que no sean a través de canales oficiales de comunicación."
]
Ejemplos Negativos:
[
    "No hay restricciones específicas sobre el uso de redes sociales para comunicarse con personal del colegio.",
    "Uso de redes sociales permitido para comunicarse con docentes bajo ciertas condiciones.",
    "No se mencionan prohibiciones explícitas sobre el contacto en redes sociales."
]
Tareas:
1. **Identificación de Acápites Relacionados**:
   - Utiliza funciones de búsqueda (por ejemplo, Ctrl + F) para identificar los principales acápites referentes al uso de celulares, dispositivos móviles, redes sociales, comunicación y sus sanciones asociadas.
   - Prioriza los siguientes términos de búsqueda: redes, sociales, facebook, whatsapp, instagram, twitter, celular(es), teléfonos móviles, tablet, smartphone, dispositivo(s), tecnológico(s)/tecnología, ciberacoso y ciberbullying (considerando que a veces el prefijo ciber- es escrito cyber-).

2. **Análisis y Codificación**:
   - Analiza cada acápite identificado y determina si la categoría "Prohíbe Contacto con el Establecimiento Vía Redes Sociales" le corresponde, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de la categoría.
Output: Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, y responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.""", "PROHIBE_REDES"),

    ("SIN_REGULACION", """Rol: Eres experto en análisis cualitativo.
Objetivo: Debes hacer una codificación deductiva de manuales de convivencia escolar de Chile considerando la categoría que se indica a continuación.
Input: Manual de convivencia escolar de un colegio de Chile en formato PDF.
Categoría:
[
    {
        "Nombre": "Prohíbe Uso Durante Horario Escolar",
        "Definición": "Normativas que prohíben de manera absoluta el uso de dispositivos móviles durante toda la jornada escolar. Esto incluye no permitir que los estudiantes lleven dispositivos móviles al colegio, y prohibir su uso en clases, recreos, almuerzos y cualquier otra actividad escolar.",
        "Nombre Variable": "PROHIBE_JORNADA",
        "Pregunta Guía": "¿Se prohíbe explícitamente llevar o utilizar dispositivos móviles o tecnología en el establecimiento educacional, incluyendo la prohibición de su uso en clases, recreo, almuerzo y en otras actividades escolares?",
        "Consideraciones": "Aplicable solo si la prohibición es total y no se mencionan excepciones en el documento. Esta es la restricción más estricta y abarca todas las situaciones posibles durante la jornada escolar. Es excluyente con las categorías 'PROHIBE_CLASES', 'RESTRINGE_USO' y 'SIN_REGULACIONES'"
    },
    {
        "Nombre": "Prohíbe Uso Durante Clases",
        "Definición": "Normativas que prohíben de manera absoluta el uso de dispositivos móviles exclusivamente durante las sesiones de clase. Los estudiantes pueden llevar los dispositivos al colegio y usarlos en otras actividades como recreos y almuerzos, pero no pueden usarlos en el aula durante el horario de clase.",
        "Nombre Variable": "PROHIBE_CLASES",
        "Pregunta Guía": "¿Se prohíbe explícitamente el uso de dispositivo móviles, celulares o tecnología solamente en clases o aula y no en el resto de la jornada, recreo o almuerzo?",
        "Consideraciones": "Aplicable solo si la prohibición es total dentro del aula y no permite excepciones durante las sesiones de clase. Esta categoría es menos estricta que 'Prohíbe Uso Durante Horario Escolar'. Es excluyente con las categorías 'PROHIBE_JORNADA', 'RESTRINGE_USO' y 'SIN_REGULACIONES'"
    },
    {
        "Nombre": "Restringe Uso Durante Clases",
        "Definición": "Normativas que restringen el uso de dispositivos móviles durante las clases pero permiten ciertas excepciones. Estas normativas no prohíben el uso de dispositivos móviles de manera absoluta ni durante toda la jornada escolar ni durante las sesiones de clase.",
        "Nombre Variable": "RESTRINGE_USO",
        "Pregunta Guía": "¿Existen excepciones específicas que autorizan el uso de dispositivos móviles y tecnología en clases?",
        "Consideraciones": "Aplicable si el documento permite el uso de dispositivos móviles bajo ciertas condiciones o excepciones dentro del aula y durante las clases. Esta categoría es más flexible y permite ciertos usos autorizados. Es excluyente con las categorías 'PROHIBE_CLASES', 'PROHIBE_JORNADA' y 'SIN_REGULACIONES'"
    },
    {
        "Nombre": "Sin Regulación",
        "Definición": "Indica que no se encontraron regulaciones o normas específicas relacionadas con el uso de dispositivos móviles y tecnología en el documento revisado. No hay restricciones ni limitaciones sobre el uso de estos dispositivos.",
        "Nombre Variable": "SIN_REGULACION",
        "Pregunta Guía": "¿No existen secciones que regulen el uso de dispositivos móviles o tecnología?",
        "Consideraciones": "Aplicable si el documento carece completamente de normas o regulaciones sobre el uso de dispositivos móviles, indicando una ausencia total de restricciones. Es excluyente con las categorías 'PROHIBE_CLASES', 'RESTRINGE_USO' y 'PROHIBE_CLASES'"
    }
]
Ejemplos Positivos:
[
    "No se encontraron regulaciones específicas sobre el uso de dispositivos móviles.",
    "El documento no menciona normas sobre el uso de teléfonos celulares.",
    "No hay regulaciones explícitas sobre dispositivos móviles en el documento.",
    "El uso de tecnología móvil no está regulado según el documento revisado.",
    "No existen secciones que regulen el uso de dispositivos móviles en el documento."
]
Ejemplos Negativos:
[
    "Uso de dispositivos móviles estrictamente regulado durante toda la jornada escolar.",
    "Normas claras sobre el uso de dispositivos móviles en el documento.",
    "Documento incluye regulaciones explícitas sobre el uso de teléfonos celulares."
]
Tareas:
1. **Identificación de Acápites Relacionados**:
   - Utiliza funciones de búsqueda (por ejemplo, Ctrl + F) para identificar los principales acápites referentes al uso de celulares, dispositivos móviles y sus sanciones asociadas.
   - Prioriza los siguientes términos de búsqueda: celular(es), teléfonos móviles, tablet, smartphone, dispositivo(s), tecnológico(s)/tecnología, ciberacoso y ciberbullying (considerando que a veces el prefijo ciber- es escrito cyber-).

2. **Análisis y Codificación**:
   - Analiza cada acápite identificado y determina si la categoría "Sin Regulación" le corresponde, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de la categoría.
Output: Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, y responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.""", "SIN_REGULACION"),

    ("PROHIBE_LABORAL", """Rol: Eres experto en análisis cualitativo.
Objetivo: Debes hacer una codificación deductiva de manuales de convivencia escolar de Chile considerando la categoría que se indica a continuación.
Input: Manual de convivencia escolar de un colegio de Chile en formato PDF.
Categoría:
[
    {
        "Nombre": "Prohíbe Uso Horario Laboral",
        "Definición": "Normativas que regulan el uso de dispositivos móviles por parte del personal docente durante el horario laboral, enfocadas en mantener la profesionalidad y evitar distracciones.",
        "Nombre Variable": "PROHIBE_LABORAL",
        "Pregunta Guía": "¿Se prohíbe al personal docente el uso de dispositivos durante la jornada laboral (prohibición dirigida al personal)?",
        "Consideraciones": "Aplicable si el documento menciona restricciones específicas al uso de dispositivos móviles por parte del personal docente o trabajadores durante la jornada laboral."
    }
]
Ejemplos Positivos:
[
    "No hablar con celular durante el desarrollo de clases u otra actividad educativa y, menos aún salir de la sala de clases para hacerlo, descuidando la protección y observación vigilante de las estudiantes, que están bajo su responsabilidad, salvo que se trate de alguna emergencia o se utilice con fines pedagógicos.",
    "El uso de instrumentos tecnológicos, tanto para estudiantes, profesores/as, directivos o asistentes de la educación está regulado.",
    "Quedan prohibidas expresamente, para cualquier funcionario del centro educativo, utilizar el teléfono celular mientras se desarrollan actividades con los estudiantes con fines pedagógicos.",
    "El (la) profesor (a) debe mantener apagado su celular mientras realiza las clases.",
    "No usar celulares durante las horas de clases frente a sus alumnos y alumnas. Así como tampoco subir a alguna red social algo que atente a la integridad de nuestra comunidad educativa."
]
Ejemplos Negativos:
[
    "El uso de dispositivos móviles por parte del personal docente está permitido en ciertas circunstancias.",
    "No se mencionan restricciones específicas sobre el uso de celulares por parte del personal docente.",
    "Para promover prácticas pedagógicas que faciliten el aprendizaje de los estudiantes, está permitido durante la hora de clases la utilización de aparatos electrónicos, tales como: teléfonos celulares, cámaras fotográficas, aparatos reproductores de música, computadores, tablets, entre otros."
]
Tareas:
1. **Identificación de Acápites Relacionados**:
   - Utiliza funciones de búsqueda (por ejemplo, Ctrl + F) para identificar los principales acápites referentes al uso de celulares, dispositivos móviles y sus sanciones asociadas para docentes y personal del colegio.
   - Prioriza los siguientes términos de búsqueda: llamadas, contestar, docentes, personal, celular(es), teléfonos móviles, tablet, smartphone, dispositivo(s), tecnológico(s)/tecnología, ciberacoso y ciberbullying (considerando que a veces el prefijo ciber- es escrito cyber-).

2. **Análisis y Codificación**:
   - Analiza cada acápite identificado y determina si la categoría "Prohíbe Uso Horario Laboral" le corresponde, considerando la definición, ejemplos positivos, ejemplos negativos, pregunta guía y consideraciones de la categoría.
Output: Responde solo indicando con un 1 en caso que la categoría se ajuste a lo indicado en el documento, y responde con un 0 en caso contrario, no incluyas nada más en tu respuesta.""", "PROHIBE_LABORAL")
]

def create_vector_db_for_document(doc_id):
    pdf_path = f"./muestra/ReglamentoConvivencia_{doc_id}.pdf"
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
   
    if not documents:
        raise ValueError(f"No se pudieron cargar documentos del archivo {pdf_path}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)

    if not texts:
        raise ValueError(f"No se pudo dividir en texto el documento {pdf_path}") 

    vector_db_path = os.path.join(VECTOR_DB_DIR, f"vector_db_{doc_id}.faiss")
    metadata_path = os.path.join(METADATA_DIR, f"metadata_{doc_id}.pkl")
    
    if not os.path.exists(vector_db_path) or not os.path.exists(metadata_path):
        index = FAISS.from_documents(texts, embeddings)
        index.save_local(vector_db_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump({'document_id': doc_id}, f)
    else:
        index = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    
    return index, texts

# Asegurarse de que los directorios existen
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# Definir el nombre del archivo CSV
csv_filename = "responses_with_categories_1.csv"

# Verificar si el archivo CSV ya existe y leer los datos existentes
if os.path.exists(csv_filename):
    existing_df = pd.read_csv(csv_filename)
    existing_results = existing_df.to_dict('records')
else:
    existing_results = []

# Inicializar la lista de resultados
results = existing_results

# Procesar cada documento
for doc_id in document_ids:
    try:
        print(f"Procesando documento {doc_id}...")
        index, texts = create_vector_db_for_document(doc_id)
        retriever = index.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        # Realizar las consultas
        responses = {}
        for category, prompt, code in prompts:
            print(f"Consultando documento {doc_id} para la categoría {category}...")
            llm_response = qa({"query": prompt, "documents": texts})
            response = llm_response["result"]
            responses[code] = response.strip()
        
    except Exception as err:
        responses = {code: f'Exception occurred. Please try again: {str(err)}' for _, _, code in prompts}
    
    # Guardar el resultado actual
    results.append({'id': doc_id, **responses})
    
    # Guardar los resultados intermedios en un archivo CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_filename, index=False)

print(f"Results saved to {csv_filename}")