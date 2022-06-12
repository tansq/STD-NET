
import sys
# sys.path.append('./snl912_c64/')

import SRNetL9L12C64 as snc64
import SRNetL9L12C64_2 as snc64_2
import SRNetL9L12C64_3 as snc64_3
import SRNetL9L12C64_32 as snc64_32
import SRNetL9L12C64_4 as snc64_4
import SRNetL9L12C64_42 as snc64_42
import SRNetL9L12C64_5 as snc64_5
import SRNetL9L12C64_52 as snc64_52
import SRNetL9L12C64_6 as snc64_6
import SRNetL9L12C64_62 as snc64_62
import SRNetL9L12C64_7 as snc64_7
import SRNetL9L12C64_72 as snc64_72
import SRNetL9L12C64_8 as snc64_8
import SRNetL9L12C64_82 as snc64_82
import SRNetL9L12C64_9 as snc64_9
import SRNetL9L12C64_92 as snc64_92
import SRNetL9L12C64_10 as snc64_10
import SRNetL9L12C64_102 as snc64_102
import SRNetL9L12C64_11 as snc64_11
import SRNetL9L12C64_112 as snc64_112
import SRNetL9L12C64_12 as snc64_12
import SRNetL9L12C64_122 as snc64_122

import SRNetDecom as snd

import SRNet as srn_ori


def choose_model_class(layer):
    layer_model = {
        21 : snc64_2.SRNetC64DE2,
        31 : snc64_3.SRNetC64DE3,
        32 : snc64_32.SRNetC64DE32,
        41 : snc64_4.SRNetC64DE4,
        42 : snc64_42.SRNetC64DE42,
        51 : snc64_5.SRNetC64DE5,
        52 : snc64_52.SRNetC64DE52,
        61 : snc64_6.SRNetC64DE6,
        62 : snc64_62.SRNetC64DE62,
        71 : snc64_7.SRNetC64DE7,
        72 : snc64_72.SRNetC64DE72,
        81 : snc64_8.SRNetC64DE8,
        82 : snc64_82.SRNetC64DE82,
        91 : snc64_9.SRNetC64DE9,
        92 : snc64_92.SRNetC64DE92,
        101 : snc64_10.SRNetC64DE10,
        102 : snc64_102.SRNetC64DE102,
        111 : snc64_11.SRNetC64DE11,
        112 : snc64_112.SRNetC64DE112,
        121 : snc64_12.SRNetC64DE12,
        122 : snc64_122.SRNetC64DE122
    }
    return layer_model.get(layer, None)