from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
from scipy.signal import resample_poly

from vita49io.defaults.default_payload_formats import DefaultPayloadFormats
from vita49io.io.iq_writer import IQStreamWriter
from vita49io.io.packet_reader import PacketReader
from vita49io.io.payload_codec import payload_as_numpy
from vita49io.protocol.context_packet import ContextPacket
from vita49io.protocol.data_packet import DataPacket

SUPPORTED_INPUT_SAMPLE_RATES = {
    98_304_000,
    24_576_000,
}

SUPPORTED_OUTPUT_SAMPLE_RATES = {
    24_576_000,
    12_288_000,
}

OUTPUT_BANDWIDTH_BY_SAMPLE_RATE = {
    24_576_000: 21_000_000,
    12_288_000: 10_500_000,
}

# Replace the None values with your designed filter taps for each rate conversion.
# Example: np.array([...], dtype=np.float64)
FILTER_TAPS_BY_RATE: Dict[Tuple[int, int], Optional[Any]] = {
    (98_304_000, 24_576_000): [ 3.50120161e-05,  4.45226230e-05,  3.73608521e-05, -1.62015978e-05, -1.30550125e-04, -2.99093910e-04, -4.87749660e-04, -6.40193570e-04, -6.95808576e-04, -6.15175507e-04, -4.02468819e-04, -1.12918167e-04,  1.61702925e-04,  3.26751272e-04,  3.25014121e-04,  1.65407865e-04, -7.53511712e-05, -2.82570684e-04, -3.54102467e-04, -2.49377747e-04, -1.27286312e-05,  2.43027174e-04,  3.88415592e-04,  3.41560735e-04,  1.13449454e-04, -1.90633288e-04, -4.18012044e-04, -4.44008714e-04, -2.39093618e-04,  1.07853570e-04,  4.26260734e-04,  5.46483469e-04,  3.89035734e-04,  1.39657615e-05, -3.99034650e-04, -6.35368698e-04, -5.55866642e-04, -1.76629669e-04,  3.26593269e-04,  6.98146447e-04,  7.31197193e-04,  3.81188202e-04, -1.97365875e-04, -7.16680335e-04, -8.97504751e-04, -6.17853648e-04,  9.90434346e-06,  6.80237588e-04,  1.04149794e-03,  8.78908341e-04,  2.39066073e-04, -5.74872471e-04, -1.14465712e-03, -1.15034162e-03, -5.47481598e-04,  3.88809388e-04,  1.18643319e-03,  1.41259594e-03,  9.06539763e-04, -1.15452063e-04, -1.14788050e-03, -1.64396186e-03, -1.30315369e-03, -2.48164552e-04,  1.01063896e-03,  1.81940074e-03,  1.71827401e-03,  6.99094851e-04, -7.59346939e-04, -1.91258538e-03, -2.12806893e-03, -1.22887187e-03,  3.81705911e-04,  1.89566244e-03,  2.50277845e-03,  1.82106746e-03,  1.28127341e-04, -1.74295697e-03, -2.80991860e-03, -2.45367232e-03, -7.71115218e-04,  1.42968488e-03,  3.01278926e-03,  3.09695751e-03,  1.54068699e-03, -9.34621486e-04, -3.07230957e-03, -3.71398084e-03, -2.42180191e-03,  2.41982330e-04,  2.94920729e-03,  4.26238664e-03,  3.39213564e-03,  6.59252932e-04, -2.60337344e-03, -4.69349053e-03, -4.42076802e-03, -1.77390941e-03,  1.99514005e-03,  4.95315274e-03,  5.46876800e-03,  3.10111513e-03, -1.08460786e-03, -4.98106895e-03, -6.48879495e-03, -4.63464389e-03, -1.69290456e-04,  4.70999571e-03,  7.42576204e-03,  6.36585732e-03,  1.81370175e-03, -4.06088411e-03, -8.21485846e-03, -8.28676328e-03, -3.90939074e-03,  2.93470105e-03,  8.77870758e-03,  1.03971380e-02,  6.54733914e-03, -1.19410533e-03, -9.01997031e-03, -1.27189291e-02, -9.88517083e-03, -1.37934579e-03,  8.80043467e-03,  1.53238009e-02,  1.42276483e-02,  5.18512121e-03, -7.89018327e-03, -1.84042316e-02, -2.02429104e-02, -1.10859735e-02,  5.81306319e-03,  2.24870678e-02,  2.96694346e-02,  2.14035729e-02, -1.25154848e-03, -2.93241792e-02, -4.86283668e-02, -4.52589734e-02, -1.17303448e-02,  4.88749319e-02,  1.22715055e-01,  1.89346795e-01,  2.28792884e-01,  2.28792884e-01,  1.89346795e-01,  1.22715055e-01,  4.88749319e-02, -1.17303448e-02, -4.52589734e-02, -4.86283668e-02, -2.93241792e-02, -1.25154848e-03,  2.14035729e-02,  2.96694346e-02,  2.24870678e-02,  5.81306319e-03, -1.10859735e-02, -2.02429104e-02, -1.84042316e-02, -7.89018327e-03,  5.18512121e-03,  1.42276483e-02,  1.53238009e-02,  8.80043467e-03, -1.37934579e-03, -9.88517083e-03, -1.27189291e-02, -9.01997031e-03, -1.19410533e-03,  6.54733914e-03,  1.03971380e-02,  8.77870758e-03,  2.93470105e-03, -3.90939074e-03, -8.28676328e-03, -8.21485846e-03, -4.06088411e-03,  1.81370175e-03,  6.36585732e-03,  7.42576204e-03,  4.70999571e-03, -1.69290456e-04, -4.63464389e-03, -6.48879495e-03, -4.98106895e-03, -1.08460786e-03,  3.10111513e-03,  5.46876800e-03,  4.95315274e-03,  1.99514005e-03, -1.77390941e-03, -4.42076802e-03, -4.69349053e-03, -2.60337344e-03,  6.59252932e-04,  3.39213564e-03,  4.26238664e-03,  2.94920729e-03,  2.41982330e-04, -2.42180191e-03, -3.71398084e-03, -3.07230957e-03, -9.34621486e-04,  1.54068699e-03,  3.09695751e-03,  3.01278926e-03,  1.42968488e-03, -7.71115218e-04, -2.45367232e-03, -2.80991860e-03, -1.74295697e-03,  1.28127341e-04,  1.82106746e-03,  2.50277845e-03,  1.89566244e-03,  3.81705911e-04, -1.22887187e-03, -2.12806893e-03, -1.91258538e-03, -7.59346939e-04,  6.99094851e-04,  1.71827401e-03,  1.81940074e-03,  1.01063896e-03, -2.48164552e-04, -1.30315369e-03, -1.64396186e-03, -1.14788050e-03, -1.15452063e-04,  9.06539763e-04,  1.41259594e-03,  1.18643319e-03,  3.88809388e-04, -5.47481598e-04, -1.15034162e-03, -1.14465712e-03, -5.74872471e-04,  2.39066073e-04,  8.78908341e-04,  1.04149794e-03,  6.80237588e-04,  9.90434346e-06, -6.17853648e-04, -8.97504751e-04, -7.16680335e-04, -1.97365875e-04,  3.81188202e-04,  7.31197193e-04,  6.98146447e-04,  3.26593269e-04, -1.76629669e-04, -5.55866642e-04, -6.35368698e-04, -3.99034650e-04,  1.39657615e-05,  3.89035734e-04,  5.46483469e-04,  4.26260734e-04,  1.07853570e-04, -2.39093618e-04, -4.44008714e-04, -4.18012044e-04, -1.90633288e-04,  1.13449454e-04,  3.41560735e-04,  3.88415592e-04,  2.43027174e-04, -1.27286312e-05, -2.49377747e-04, -3.54102467e-04, -2.82570684e-04, -7.53511712e-05,  1.65407865e-04,  3.25014121e-04,  3.26751272e-04,  1.61702925e-04, -1.12918167e-04, -4.02468819e-04, -6.15175507e-04, -6.95808576e-04, -6.40193570e-04, -4.87749660e-04, -2.99093910e-04, -1.30550125e-04, -1.62015978e-05,  3.73608521e-05,  4.45226230e-05,  3.50120161e-05],
    (98_304_000, 12_288_000): [ 2.15366677e-05,  1.25311937e-05,  1.25173051e-05,  8.77296293e-06, -4.82919058e-08, -1.51193977e-05, -3.72557864e-05, -6.67094310e-05, -1.02952368e-04, -1.44648951e-04, -1.89650279e-04, -2.34908949e-04, -2.76937607e-04, -3.11855504e-04, -3.35916940e-04, -3.45862430e-04, -3.39332654e-04, -3.15236373e-04, -2.74019377e-04, -2.17771251e-04, -1.50156162e-04, -7.62169631e-05, -1.87791115e-06,  6.65316274e-05,  1.22989050e-04,  1.62403004e-04,  1.81238564e-04,  1.77969176e-04,  1.53366390e-04,  1.10494616e-04,  5.44786894e-05, -8.00007054e-06, -6.94329068e-05, -1.22309914e-04, -1.60022008e-04, -1.77673347e-04, -1.72721991e-04, -1.45370462e-04, -9.86248926e-05, -3.80143378e-05,  2.90233929e-05,  9.40023891e-05,  1.48453523e-04,  1.84978789e-04,  1.98265435e-04,  1.85825201e-04,  1.48435533e-04,  9.01470622e-05,  1.78960814e-05, -5.93059069e-05, -1.31454396e-04, -1.88833990e-04, -2.23302450e-04, -2.29424389e-04, -2.05297751e-04, -1.52949612e-04, -7.82220510e-05,  9.83983015e-06,  1.00061693e-04,  1.80531672e-04,  2.40156397e-04,  2.70161730e-04,  2.65367315e-04,  2.25009005e-04,  1.53038789e-04,  5.77780201e-05, -4.90007837e-05, -1.53514221e-04, -2.41714970e-04, -3.01170753e-04, -3.22780401e-04, -3.02138171e-04, -2.40308514e-04, -1.43910874e-04, -2.44567676e-05,  1.02990311e-04,  2.21694785e-04,  3.15419608e-04,  3.70628985e-04,  3.78408750e-04,  3.35850953e-04,  2.46679398e-04,  1.21020695e-04, -2.57104240e-05, -1.74661672e-04, -3.05949930e-04, -4.01283697e-04, -4.46491900e-04, -4.33581929e-04, -3.62063880e-04, -2.39298148e-04, -7.98171994e-05,  9.63601387e-05,  2.66207446e-04,  4.06677772e-04,  4.97801755e-04,  5.25496207e-04,  4.83678571e-04,  3.75391182e-04,  2.12731174e-04,  1.55619845e-05, -1.90875486e-04, -3.79160362e-04, -5.23311255e-04, -6.02334682e-04, -6.03244235e-04, -5.23098752e-04, -3.69770542e-04, -1.61272077e-04,  7.62998564e-05,  3.11998031e-04,  5.14018564e-04,  6.53965642e-04,  7.10800967e-04,  6.73947101e-04,  5.45084514e-04,  3.38360979e-04,  7.89198253e-05, -2.00104774e-04, -4.61781782e-04, -6.70235541e-04, -7.95526927e-04, -8.17901371e-04, -7.30811669e-04, -5.42255965e-04, -2.74202120e-04,  3.99020976e-05,  3.59251010e-04,  6.40944165e-04,  8.45707740e-04,  9.43327137e-04,  9.17033160e-04,  7.66225619e-04,  5.07097970e-04,  1.71015111e-04, -1.99216783e-04, -5.54752251e-04, -8.47107557e-04, -1.03472583e-03, -1.08882482e-03, -9.97712112e-04, -7.68930214e-04, -4.28872942e-04, -1.98376930e-05,  4.05170781e-04,  7.89255440e-04,  1.07919371e-03,  1.23275467e-03,  1.22476907e-03,  1.05108893e-03,  7.29833213e-04,  2.99659325e-04, -1.84798586e-04, -6.59802615e-04, -1.06083882e-03, -1.33131832e-03, -1.43048915e-03, -1.33945848e-03, -1.06445284e-03, -6.36781587e-04, -1.09407937e-04,  4.49524826e-04,  9.65424575e-04,  1.36713892e-03,  1.59669127e-03,  1.61754860e-03,  1.42024900e-03,  1.02454834e-03,  4.77678033e-04, -1.51168139e-04, -7.79498563e-04, -1.32226268e-03, -1.70328077e-03, -1.86583618e-03, -1.78097423e-03, -1.45230525e-03, -9.16557906e-04, -2.39663709e-04,  4.91231462e-04,  1.17888686e-03,  1.72880321e-03,  2.06210505e-03,  2.12671490e-03,  1.90526604e-03,  1.41856569e-03,  7.24009468e-04, -9.10117984e-05, -9.19971213e-04, -1.65101035e-03, -2.18190591e-03, -2.43422248e-03, -2.36470135e-03, -1.97226422e-03, -1.29953425e-03, -4.28498972e-04,  5.29299212e-04,  1.44688686e-03,  2.19854099e-03,  2.67684595e-03,  2.80784418e-03,  2.56215833e-03,  1.96046134e-03,  1.07235345e-03,  8.59276469e-06, -1.09250820e-03, -2.08286420e-03, -2.82433003e-03, -3.20773451e-03, -3.16865569e-03, -2.69768325e-03, -1.84360668e-03, -7.08843155e-04,  5.62535155e-04,  1.80282705e-03,  2.84255103e-03,  3.53331528e-03,  3.76864957e-03,  3.49993005e-03,  2.74506280e-03,  1.58850734e-03,  1.72311093e-04, -1.32095256e-03, -2.69135607e-03, -3.74799140e-03, -4.33513004e-03, -4.35462742e-03, -3.78142995e-03, -2.66982344e-03, -1.14923991e-03,  5.90231802e-04,  2.32107587e-03,  3.80755867e-03,  4.83722494e-03,  5.25055910e-03,  4.96475083e-03,  3.98814865e-03,  2.42306901e-03,  4.56106846e-04, -1.66329767e-03, -3.65366540e-03, -5.23796116e-03, -6.18112351e-03, -6.32369569e-03, -5.60688963e-03, -4.08536826e-03, -1.92546588e-03,  6.11589420e-04,  3.19995132e-03,  5.48955989e-03,  7.15183785e-03,  7.92489618e-03,  7.65227286e-03,  6.30976470e-03,  4.01621370e-03,  1.02597511e-03, -2.29692859e-03, -5.52060776e-03, -8.19850443e-03, -9.92774428e-03, -1.04053894e-02, -9.47523159e-03, -7.15855387e-03, -3.66396762e-03,  6.26220432e-04,  5.19284998e-03,  9.43681434e-03,  1.27502241e-02,  1.45939657e-02,  1.45725709e-02,  1.24971597e-02,  8.42814412e-03,  2.69136290e-03, -4.13587968e-03, -1.12690419e-02, -1.77875077e-02, -2.27260846e-02, -2.51776169e-02, -2.43958257e-02, -1.98870200e-02, -1.14801125e-02,  6.33597821e-04,  1.58963850e-02,  3.34175629e-02,  5.20422040e-02,  7.04489816e-02,  8.72677397e-02,  1.01205092e-01,  1.11165299e-01,  1.16353994e-01,  1.16353994e-01,  1.11165299e-01,  1.01205092e-01,  8.72677397e-02,  7.04489816e-02,  5.20422040e-02,  3.34175629e-02,  1.58963850e-02,  6.33597821e-04, -1.14801125e-02, -1.98870200e-02, -2.43958257e-02, -2.51776169e-02, -2.27260846e-02, -1.77875077e-02, -1.12690419e-02, -4.13587968e-03,  2.69136290e-03,  8.42814412e-03,  1.24971597e-02,  1.45725709e-02,  1.45939657e-02,  1.27502241e-02,  9.43681434e-03,  5.19284998e-03,  6.26220432e-04, -3.66396762e-03, -7.15855387e-03, -9.47523159e-03, -1.04053894e-02, -9.92774428e-03, -8.19850443e-03, -5.52060776e-03, -2.29692859e-03,  1.02597511e-03,  4.01621370e-03,  6.30976470e-03,  7.65227286e-03,  7.92489618e-03,  7.15183785e-03,  5.48955989e-03,  3.19995132e-03,  6.11589420e-04, -1.92546588e-03, -4.08536826e-03, -5.60688963e-03, -6.32369569e-03, -6.18112351e-03, -5.23796116e-03, -3.65366540e-03, -1.66329767e-03,  4.56106846e-04,  2.42306901e-03,  3.98814865e-03,  4.96475083e-03,  5.25055910e-03,  4.83722494e-03,  3.80755867e-03,  2.32107587e-03,  5.90231802e-04, -1.14923991e-03, -2.66982344e-03, -3.78142995e-03, -4.35462742e-03, -4.33513004e-03, -3.74799140e-03, -2.69135607e-03, -1.32095256e-03,  1.72311093e-04,  1.58850734e-03,  2.74506280e-03,  3.49993005e-03,  3.76864957e-03,  3.53331528e-03,  2.84255103e-03,  1.80282705e-03,  5.62535155e-04, -7.08843155e-04, -1.84360668e-03, -2.69768325e-03, -3.16865569e-03, -3.20773451e-03, -2.82433003e-03, -2.08286420e-03, -1.09250820e-03,  8.59276469e-06,  1.07235345e-03,  1.96046134e-03,  2.56215833e-03,  2.80784418e-03,  2.67684595e-03,  2.19854099e-03,  1.44688686e-03,  5.29299212e-04, -4.28498972e-04, -1.29953425e-03, -1.97226422e-03, -2.36470135e-03, -2.43422248e-03, -2.18190591e-03, -1.65101035e-03, -9.19971213e-04, -9.10117984e-05,  7.24009468e-04,  1.41856569e-03,  1.90526604e-03,  2.12671490e-03,  2.06210505e-03,  1.72880321e-03,  1.17888686e-03,  4.91231462e-04, -2.39663709e-04, -9.16557906e-04, -1.45230525e-03, -1.78097423e-03, -1.86583618e-03, -1.70328077e-03, -1.32226268e-03, -7.79498563e-04, -1.51168139e-04,  4.77678033e-04,  1.02454834e-03,  1.42024900e-03,  1.61754860e-03,  1.59669127e-03,  1.36713892e-03,  9.65424575e-04,  4.49524826e-04, -1.09407937e-04, -6.36781587e-04, -1.06445284e-03, -1.33945848e-03, -1.43048915e-03, -1.33131832e-03, -1.06083882e-03, -6.59802615e-04, -1.84798586e-04,  2.99659325e-04,  7.29833213e-04,  1.05108893e-03,  1.22476907e-03,  1.23275467e-03,  1.07919371e-03,  7.89255440e-04,  4.05170781e-04, -1.98376930e-05, -4.28872942e-04, -7.68930214e-04, -9.97712112e-04, -1.08882482e-03, -1.03472583e-03, -8.47107557e-04, -5.54752251e-04, -1.99216783e-04,  1.71015111e-04,  5.07097970e-04,  7.66225619e-04,  9.17033160e-04,  9.43327137e-04,  8.45707740e-04,  6.40944165e-04,  3.59251010e-04,  3.99020976e-05, -2.74202120e-04, -5.42255965e-04, -7.30811669e-04, -8.17901371e-04, -7.95526927e-04, -6.70235541e-04, -4.61781782e-04, -2.00104774e-04,  7.89198253e-05,  3.38360979e-04,  5.45084514e-04,  6.73947101e-04,  7.10800967e-04,  6.53965642e-04,  5.14018564e-04,  3.11998031e-04,  7.62998564e-05, -1.61272077e-04, -3.69770542e-04, -5.23098752e-04, -6.03244235e-04, -6.02334682e-04, -5.23311255e-04, -3.79160362e-04, -1.90875486e-04,  1.55619845e-05,  2.12731174e-04,  3.75391182e-04,  4.83678571e-04,  5.25496207e-04,  4.97801755e-04,  4.06677772e-04,  2.66207446e-04,  9.63601387e-05, -7.98171994e-05, -2.39298148e-04, -3.62063880e-04, -4.33581929e-04, -4.46491900e-04, -4.01283697e-04, -3.05949930e-04, -1.74661672e-04, -2.57104240e-05,  1.21020695e-04,  2.46679398e-04,  3.35850953e-04,  3.78408750e-04,  3.70628985e-04,  3.15419608e-04,  2.21694785e-04,  1.02990311e-04, -2.44567676e-05, -1.43910874e-04, -2.40308514e-04, -3.02138171e-04, -3.22780401e-04, -3.01170753e-04, -2.41714970e-04, -1.53514221e-04, -4.90007837e-05,  5.77780201e-05,  1.53038789e-04,  2.25009005e-04,  2.65367315e-04,  2.70161730e-04,  2.40156397e-04,  1.80531672e-04,  1.00061693e-04,  9.83983015e-06, -7.82220510e-05, -1.52949612e-04, -2.05297751e-04, -2.29424389e-04, -2.23302450e-04, -1.88833990e-04, -1.31454396e-04, -5.93059069e-05,  1.78960814e-05,  9.01470622e-05,  1.48435533e-04,  1.85825201e-04,  1.98265435e-04,  1.84978789e-04,  1.48453523e-04,  9.40023891e-05,  2.90233929e-05, -3.80143378e-05, -9.86248926e-05, -1.45370462e-04, -1.72721991e-04, -1.77673347e-04, -1.60022008e-04, -1.22309914e-04, -6.94329068e-05, -8.00007054e-06,  5.44786894e-05,  1.10494616e-04,  1.53366390e-04,  1.77969176e-04,  1.81238564e-04,  1.62403004e-04,  1.22989050e-04,  6.65316274e-05, -1.87791115e-06, -7.62169631e-05, -1.50156162e-04, -2.17771251e-04, -2.74019377e-04, -3.15236373e-04, -3.39332654e-04, -3.45862430e-04, -3.35916940e-04, -3.11855504e-04, -2.76937607e-04, -2.34908949e-04, -1.89650279e-04, -1.44648951e-04, -1.02952368e-04, -6.67094310e-05, -3.72557864e-05, -1.51193977e-05, -4.82919058e-08,  8.77296293e-06,  1.25173051e-05,  1.25311937e-05,  2.15366677e-05],
    (24_576_000, 12_288_000): [ 7.89457737e-05,  3.90665846e-05, -4.12034647e-04, -1.13909908e-03, -1.34664409e-03, -5.28478502e-04,  5.22090667e-04,  5.23205227e-04, -3.79516977e-04, -6.46253659e-04,  2.41476318e-04,  7.82232779e-04, -7.54304568e-05, -9.22450936e-04, -1.49738910e-04,  1.03826640e-03,  4.41241946e-04, -1.10179228e-03, -7.95017392e-04,  1.08631233e-03,  1.19818469e-03, -9.67294714e-04, -1.62977424e-03,  7.23830088e-04,  2.06137144e-03, -3.40472083e-04, -2.45788736e-03, -1.90922146e-04,  2.77877981e-03,  8.69436448e-04, -2.97974982e-03, -1.68357066e-03,  3.01486794e-03,  2.61009299e-03, -2.83904460e-03, -3.61338052e-03,  2.41070113e-03,  4.64536781e-03, -1.69457481e-03, -5.64601362e-03,  6.64394777e-04,  6.54429149e-03,  6.94737722e-04, -7.25960977e-03, -2.38438532e-03,  7.70340172e-03,  4.39140955e-03, -7.78057811e-03, -6.68730190e-03,  7.39031027e-03,  9.22832173e-03, -6.42537714e-03, -1.19564623e-02,  4.76875409e-03,  1.48012322e-02, -2.28506690e-03, -1.76822017e-02, -1.19774463e-03,  2.05121754e-02,  5.92676093e-03, -2.32008815e-02, -1.23070597e-02,  2.56589759e-02,  2.10975965e-02, -2.78021249e-02, -3.39440753e-02,  2.95549593e-02,  5.51896946e-02, -3.08546690e-02, -1.00922832e-01,  3.16540515e-02,  3.16566668e-01,  4.68076178e-01,  3.16566668e-01,  3.16540515e-02, -1.00922832e-01, -3.08546690e-02,  5.51896946e-02,  2.95549593e-02, -3.39440753e-02, -2.78021249e-02,  2.10975965e-02,  2.56589759e-02, -1.23070597e-02, -2.32008815e-02,  5.92676093e-03,  2.05121754e-02, -1.19774463e-03, -1.76822017e-02, -2.28506690e-03,  1.48012322e-02,  4.76875409e-03, -1.19564623e-02, -6.42537714e-03,  9.22832173e-03,  7.39031027e-03, -6.68730190e-03, -7.78057811e-03,  4.39140955e-03,  7.70340172e-03, -2.38438532e-03, -7.25960977e-03,  6.94737722e-04,  6.54429149e-03,  6.64394777e-04, -5.64601362e-03, -1.69457481e-03,  4.64536781e-03,  2.41070113e-03, -3.61338052e-03, -2.83904460e-03,  2.61009299e-03,  3.01486794e-03, -1.68357066e-03, -2.97974982e-03,  8.69436448e-04,  2.77877981e-03, -1.90922146e-04, -2.45788736e-03, -3.40472083e-04,  2.06137144e-03,  7.23830088e-04, -1.62977424e-03, -9.67294714e-04,  1.19818469e-03,  1.08631233e-03, -7.95017392e-04, -1.10179228e-03,  4.41241946e-04,  1.03826640e-03, -1.49738910e-04, -9.22450936e-04, -7.54304568e-05,  7.82232779e-04,  2.41476318e-04, -6.46253659e-04, -3.79516977e-04,  5.23205227e-04,  5.22090667e-04, -5.28478502e-04, -1.34664409e-03, -1.13909908e-03, -4.12034647e-04,  3.90665846e-05,  7.89457737e-05],
    (24_576_000, 24_576_000): None,
}


def _ensure_src_on_path() -> None:
    # Allow running the example from the repo root without installation
    here = Path(__file__).resolve().parent
    src = here.parent / "src"
    src_str = str(src)
    if src.is_dir() and src_str not in sys.path:
        sys.path.insert(0, src_str)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDC a VITA 49 file (resample + re-pack).")
    parser.add_argument("input_file", help="Path to input .v49 file")
    parser.add_argument("output_file", help="Path to output .v49 file")
    parser.add_argument(
        "--output-format",
        required=True,
        help="Output payload format (F32_IQ, S32_IQ, S24_IQ, S16_IQ)",
    )
    parser.add_argument(
        "--output-sample-rate",
        required=True,
        type=int,
        help="Output sample rate in Hz",
    )
    parser.add_argument(
        "--chunk-samples",
        type=int,
        default=61_140,
        help="Input samples per resampling block",
    )
    parser.add_argument(
        "--samples-per-packet",
        type=int,
        default=1024,
        help="Number of complex samples per output data packet",
    )
    return parser.parse_args(argv)


def _packet_time_s(integer_seconds: Optional[int], fractional_seconds: Optional[int]) -> Optional[float]:
    if integer_seconds is None and fractional_seconds is None:
        return None
    sec = float(integer_seconds or 0)
    frac = float(fractional_seconds or 0)
    return sec + (frac / float(1 << 64))


def _resolve_input_format_name(pf, supported_formats: Dict[str, Any]) -> Optional[str]:
    for name, fmt in supported_formats.items():
        if pf == fmt:
            return name
    return None


def _resample_ratio(in_rate_hz: int, out_rate_hz: int) -> Tuple[int, int]:
    frac = Fraction(int(out_rate_hz), int(in_rate_hz))
    return frac.numerator, frac.denominator


def _get_resample_window(in_rate_hz: int, out_rate_hz: int):
    key = (in_rate_hz, out_rate_hz)
    if key not in FILTER_TAPS_BY_RATE:
        raise ValueError(f"Unsupported rate conversion: {in_rate_hz} -> {out_rate_hz}")
    return FILTER_TAPS_BY_RATE[key]


def convert_v49_ddc(
    input_path: Path,
    output_path: Path,
    output_format_name: str,
    output_sample_rate_hz: int,
    chunk_samples: int,
    samples_per_packet: int,
) -> Dict[str, int]:
    _ensure_src_on_path()


    supported_formats = {
        "F32_IQ": DefaultPayloadFormats.F32_IQ,
        "S32_IQ": DefaultPayloadFormats.S32_IQ,
        "S24_IQ": DefaultPayloadFormats.S24_IQ,
        "S16_IQ": DefaultPayloadFormats.S16_IQ,
    }

    output_format_name = output_format_name.upper()
    if output_format_name not in supported_formats:
        raise ValueError(
            f"Unsupported output format '{output_format_name}'. "
            f"Supported: {', '.join(sorted(supported_formats))}"
        )
    output_payload_format = supported_formats[output_format_name]

    if output_sample_rate_hz not in SUPPORTED_OUTPUT_SAMPLE_RATES:
        raise ValueError(
            "Unsupported output sample rate. Supported: "
            + ", ".join(str(x) for x in sorted(SUPPORTED_OUTPUT_SAMPLE_RATES))
        )

    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be > 0")
    if samples_per_packet <= 0:
        raise ValueError("samples_per_packet must be > 0")

    output_bandwidth_hz = OUTPUT_BANDWIDTH_BY_SAMPLE_RATE.get(output_sample_rate_hz)
    if output_bandwidth_hz is None:
        raise ValueError(
            f"No bandwidth mapping for output sample rate {output_sample_rate_hz}"
        )

    input_path = input_path.expanduser()
    output_path = output_path.expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # State from the first context packet
    input_sample_rate_hz: Optional[int] = None
    input_payload_format = None
    input_payload_name: Optional[str] = None
    input_bandwidth_hz: Optional[float] = None
    input_rf_ref_hz: Optional[float] = None
    input_rf_ref_offset_hz: Optional[float] = None
    input_if_ref_hz: Optional[float] = None
    input_if_band_offset_hz: Optional[float] = None
    input_reference_level_dbm: Optional[float] = None
    input_gain_db: Optional[Tuple[float, float]] = None
    input_device_identifier: Optional[Tuple[int, int]] = None
    input_state_event_indicators: Optional[int] = None
    input_context_tsm: bool = False
    input_stream_id: Optional[int] = None
    first_context_time_s: Optional[float] = None

    # Resampling parameters
    up: Optional[int] = None
    down: Optional[int] = None
    resample_window = None

    # Output stream state
    writer: Optional[IQStreamWriter] = None

    in_chunks: list[np.ndarray] = []
    in_count = 0
    out_buffer = np.empty(0, dtype=np.complex64)

    total_in_samples = 0
    total_out_samples = 0
    data_packets_written = 0
    data_packets_seen = 0
    context_packets_seen = 0
    skipped_packets = 0

    def emit_samples(samples: np.ndarray, out_f) -> None:
        nonlocal out_buffer, data_packets_written, total_out_samples
        if samples.size == 0:
            return
        if out_buffer.size == 0:
            out_buffer = samples
        else:
            out_buffer = np.concatenate([out_buffer, samples])
        while out_buffer.size >= samples_per_packet:
            chunk = out_buffer[:samples_per_packet]
            out_buffer = out_buffer[samples_per_packet:]
            out_f.write(writer.build_data_packet_bytes(chunk))
            data_packets_written += 1
            total_out_samples += samples_per_packet

    with input_path.open("rb") as f_in, output_path.open("wb") as f_out:
        reader = PacketReader(f_in)
        while True:
            pkt = reader.read_packet()
            if pkt is None:
                break

            if isinstance(pkt, ContextPacket):
                context_packets_seen += 1
                if first_context_time_s is None:
                    first_context_time_s = _packet_time_s(pkt.integer_seconds, pkt.fractional_seconds)
                if pkt.cif0 is not None:
                    cif0 = pkt.cif0
                    if input_payload_format is None and cif0.payload_format is not None:
                        input_payload_format = cif0.payload_format
                        input_payload_name = _resolve_input_format_name(
                            input_payload_format,
                            supported_formats,
                        )
                    if input_sample_rate_hz is None and cif0.sample_rate_hz is not None:
                        input_sample_rate_hz = int(round(float(cif0.sample_rate_hz)))
                    if input_bandwidth_hz is None and cif0.bandwidth_hz is not None:
                        input_bandwidth_hz = float(cif0.bandwidth_hz)
                    if input_rf_ref_hz is None and cif0.rf_reference_frequency_hz is not None:
                        input_rf_ref_hz = float(cif0.rf_reference_frequency_hz)
                    if input_rf_ref_offset_hz is None and cif0.rf_reference_frequency_offset_hz is not None:
                        input_rf_ref_offset_hz = float(cif0.rf_reference_frequency_offset_hz)
                    if input_if_ref_hz is None and cif0.if_reference_frequency_hz is not None:
                        input_if_ref_hz = float(cif0.if_reference_frequency_hz)
                    if input_if_band_offset_hz is None and cif0.if_band_offset_hz is not None:
                        input_if_band_offset_hz = float(cif0.if_band_offset_hz)
                    if input_reference_level_dbm is None and cif0.reference_level_dbm is not None:
                        input_reference_level_dbm = float(cif0.reference_level_dbm)
                    if input_gain_db is None and cif0.gain_db is not None:
                        input_gain_db = cif0.gain_db
                    if input_device_identifier is None and cif0.device_identifier is not None:
                        input_device_identifier = cif0.device_identifier
                    if input_state_event_indicators is None and cif0.state_event_indicators is not None:
                        input_state_event_indicators = cif0.state_event_indicators
                if input_stream_id is None and pkt.stream_id is not None:
                    input_stream_id = pkt.stream_id
                input_context_tsm = bool(pkt.header.indicators_24)
                continue

            if isinstance(pkt, DataPacket):
                data_packets_seen += 1
                if input_payload_format is None or input_sample_rate_hz is None:
                    skipped_packets += 1
                    continue
                if input_payload_name is None:
                    raise ValueError(
                        "Unsupported input payload format. "
                        "Supported: F32_IQ, S32_IQ, S24_IQ, S16_IQ"
                    )
                if input_sample_rate_hz not in SUPPORTED_INPUT_SAMPLE_RATES:
                    raise ValueError(
                        "Unsupported input sample rate. Supported: "
                        + ", ".join(str(x) for x in sorted(SUPPORTED_INPUT_SAMPLE_RATES))
                    )

                if up is None or down is None:
                    up, down = _resample_ratio(input_sample_rate_hz, output_sample_rate_hz)
                    resample_window = _get_resample_window(input_sample_rate_hz, output_sample_rate_hz)

                if writer is None:
                    if input_stream_id is None:
                        input_stream_id = pkt.stream_id
                    if input_stream_id is None:
                        raise ValueError("Input stream_id is missing; cannot write output stream")
                    start_time_s = _packet_time_s(pkt.integer_seconds, pkt.fractional_seconds)
                    if start_time_s is None:
                        start_time_s = first_context_time_s
                    writer = IQStreamWriter(
                        stream_id=input_stream_id,
                        sample_rate_hz=float(output_sample_rate_hz),
                        payload_format=output_payload_format,
                        data_packet_type=pkt.header.packet_type,
                        tsi=pkt.header.tsi,
                        tsf=pkt.header.tsf,
                        class_id=pkt.class_id,
                        requires_vita49_2=bool(pkt.header.indicators_25),
                        frequency_domain=bool(pkt.header.indicators_24),
                        start_time_epoch_s=start_time_s,
                        bandwidth_hz=float(output_bandwidth_hz),
                        rf_reference_frequency_hz=input_rf_ref_hz,
                        rf_reference_frequency_offset_hz=input_rf_ref_offset_hz,
                        if_reference_frequency_hz=input_if_ref_hz,
                        if_band_offset_hz=input_if_band_offset_hz,
                        reference_level_dbm=input_reference_level_dbm,
                        gain_db=input_gain_db,
                        device_identifier=input_device_identifier,
                        state_event_indicators=input_state_event_indicators,
                        context_timestamp_mode_general=input_context_tsm,
                    )
                    f_out.write(writer.build_context_packet().to_bytes())

                payload = pkt.payload
                payload_bytes = payload.tobytes() if isinstance(payload, memoryview) else payload
                iq = payload_as_numpy(payload_bytes, input_payload_format)

                in_chunks.append(iq)
                in_count += int(iq.size)
                total_in_samples += int(iq.size)

                while in_count >= chunk_samples:
                    combined = np.concatenate(in_chunks) if len(in_chunks) > 1 else in_chunks[0]
                    block = combined[:chunk_samples]
                    remainder = combined[chunk_samples:]
                    in_chunks = [remainder] if remainder.size else []
                    in_count = int(remainder.size)

                    if up == 1 and down == 1:
                        resampled = block
                    else:
                        resampled = resample_poly(block, up, down, window=resample_window)
                    emit_samples(np.asarray(resampled, dtype=np.complex64).reshape(-1), f_out)
                continue

            skipped_packets += 1

        # Process remaining samples after loop ends
        if in_count > 0 and input_sample_rate_hz is not None and input_payload_format is not None:
            combined = np.concatenate(in_chunks) if len(in_chunks) > 1 else in_chunks[0]
            if up is None or down is None:
                up, down = _resample_ratio(input_sample_rate_hz, output_sample_rate_hz)
                resample_window = _get_resample_window(input_sample_rate_hz, output_sample_rate_hz)
            if up == 1 and down == 1:
                resampled = combined
            else:
                resampled = resample_poly(combined, up, down, window=resample_window)
            emit_samples(np.asarray(resampled, dtype=np.complex64).reshape(-1), f_out)

        # Pad the last packet with zeros to reach samples_per_packet
        if writer is not None and out_buffer.size > 0:
            pad_len = samples_per_packet - int(out_buffer.size)
            padded = np.concatenate([out_buffer, np.zeros(pad_len, dtype=np.complex64)])
            f_out.write(writer.build_data_packet_bytes(padded))
            data_packets_written += 1
            total_out_samples += samples_per_packet

    return {
        "input_samples": total_in_samples,
        "output_samples": total_out_samples,
        "data_packets_written": data_packets_written,
        "data_packets_seen": data_packets_seen,
        "context_packets_seen": context_packets_seen,
        "skipped_packets": skipped_packets,
        "input_sample_rate_hz": int(input_sample_rate_hz or 0),
        "output_sample_rate_hz": int(output_sample_rate_hz),
        "input_payload_format": input_payload_name or "",
        "output_payload_format": output_format_name,
        "output_bandwidth_hz": int(output_bandwidth_hz),
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    try:
        summary = convert_v49_ddc(
            input_path=Path(args.input_file),
            output_path=Path(args.output_file),
            output_format_name=args.output_format,
            output_sample_rate_hz=int(args.output_sample_rate),
            chunk_samples=int(args.chunk_samples),
            samples_per_packet=int(args.samples_per_packet),
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(
        "DDC complete. "
        f"In={summary['input_sample_rate_hz']} Hz {summary['input_payload_format']} -> "
        f"Out={summary['output_sample_rate_hz']} Hz {summary['output_payload_format']} "
        f"(BW {summary['output_bandwidth_hz']}), "
        f"Input samples: {summary['input_samples']}, "
        f"Output samples: {summary['output_samples']}, "
        f"Data packets written: {summary['data_packets_written']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
