import numpy as np
import networkx as nx
import math
from sympy import Point
from Roadnet_component.Lane import Lane
from Roadnet_component.Flow import Flow
from Roadnet_component.Road import Road
from Roadnet_component.Intersection import Intersection

roadLink_type = ["turn_left", "go_straight", "turn_right"]

# origin of longitude of the flow
y_o = [
    -8.619894, -8.612955, -8.611065, -8.61255, -8.610912, -8.615394, -8.611569, -8.61147, -8.61984, -8.617041,
    -8.606403, -8.605926, -8.619849, -8.614107, -8.605368, -8.617662, -8.612478, -8.611137, -8.615736, -8.615817,
    -8.615808, -8.614341, -8.612613, -8.610876, -8.609697, -8.617707, -8.606484, -8.617572, -8.61399, -8.602848,
    -8.619867, -8.606493, -8.610831, -8.616033, -8.613846, -8.613324, -8.606547, -8.610795, -8.600508, -8.609076,
    -8.610291, -8.610858, -8.610957, -8.60949, -8.610867, -8.610912, -8.617329, -8.606484, -8.612163, -8.610912,
    -8.602425, -8.619858, -8.61984, -8.614728, -8.60877, -8.610939, -8.612892, -8.618013, -8.61417, -8.613252,
    -8.610714, -8.61984, -8.610939, -8.621046, -8.620506, -8.609607, -8.606862, -8.621001, -8.621001, -8.620992,
    -8.613216, -8.620983, -8.616402, -8.610813, -8.610867, -8.606502, -8.610966, -8.604504, -8.61093, -8.620155,
    -8.620011, -8.602101, -8.610858, -8.609463, -8.615646, -8.614044, -8.609742, -8.617743, -8.605458, -8.609391,
    -8.606313, -8.610849, -8.619885, -8.606457, -8.612595, -8.619831, -8.610804, -8.609535, -8.616366, -8.610714,
    -8.610786, -8.610957, -8.61201, -8.608788, -8.608716, -8.6139, -8.611182, -8.609679, -8.609481, -8.614287,
    -8.609508, -8.619696, -8.601948, -8.610894, -8.616105, -8.605728, -8.613207, -8.61642, -8.619849, -8.619867,
    -8.613288, -8.614296, -8.608554, -8.619831, -8.615466, -8.609877, -8.610669, -8.610867, -8.619786, -8.607141,
    -8.607024, -8.615727, -8.619777, -8.619813, -8.606475, -8.613981, -8.611137, -8.61075, -8.612532, -8.614557,
    -8.607123, -8.619822, -8.619804, -8.603424, -8.617662, -8.619831, -8.615565, -8.616141, -8.610885, -8.610768,
    -8.609256, -8.611011, -8.61552, -8.613882, -8.617545, -8.601669, -8.617482, -8.611182, -8.610219, -8.617626,
    -8.618796, -8.607132, -8.611371, -8.611047, -8.610957, -8.610822, -8.613873, -8.607105, -8.612901, -8.604765,
    -8.607654, -8.617608, -8.619858, -8.613261, -8.613225, -8.607438, -8.606799, -8.619822, -8.613873, -8.615088,
    -8.619858, -8.610831, -8.604756, -8.610912, -8.617041, -8.615538, -8.610381, -8.613918, -8.621892, -8.611056,
    -8.615457, -8.606997, -8.609967, -8.621037, -8.606583, -8.610687, -8.617653, -8.608887, -8.610876, -8.619885,
    -8.606691, -8.612154, -8.61903, -8.622135, -8.61075, -8.607636, -8.610966, -8.615448, -8.61003, -8.617572, -8.60994,
    -8.615673, -8.609454, -8.608347, -8.617752, -8.615511, -8.607348, -8.619813, -8.606592, -8.608707, -8.606448,
    -8.606484, -8.615592, -8.606322, -8.615619, -8.609679, -8.610831, -8.61975, -8.606304, -8.6067, -8.610975,
    -8.610885, -8.619624, -8.619705, -8.606448, -8.608491, -8.608941, -8.613279, -8.607105, -8.621037, -8.611164,
    -8.605782, -8.619867, -8.619813, -8.610822, -8.60742, -8.610894, -8.606475, -8.617599, -8.610777, -8.612802,
    -8.619912, -8.615565, -8.617644, -8.61984, -8.608824, -8.608887, -8.606511, -8.606538, -8.615556, -8.612955,
    -8.612532, -8.60652, -8.611209, -8.619849, -8.60904, -8.609103, -8.609067, -8.609031, -8.610831, -8.619831,
    -8.606511, -8.617626, -8.61075, -8.605719, -8.619849, -8.61246, -8.615511, -8.615538, -8.610705, -8.606484,
    -8.61354, -8.612658, -8.620695, -8.606277, -8.599275, -8.61552, -8.612946, -8.61246, -8.615556, -8.612964,
    -8.610876, -8.613, -8.618085, -8.606817, -8.611596, -8.612802, -8.606457, -8.610498, -8.615502, -8.610003, -8.61282,
    -8.620209, -8.612721, -8.60643, -8.611515, -8.612631, -8.612757, -8.612703, -8.610885, -8.61291, -8.613819,
    -8.613027, -8.615475, -8.61201, -8.613801, -8.612937, -8.606943, -8.617599, -8.613018, -8.620011, -8.617617,
    -8.619858, -8.613045, -8.617311, -8.613, -8.617698, -8.612973, -8.606403, -8.617617, -8.612748, -8.608689,
    -8.612874, -8.611119, -8.608815, -8.617599, -8.606493, -8.612703, -8.612748, -8.611515, -8.612082, -8.612982,
    -8.610669, -8.610084, -8.610138, -8.610939, -8.619741, -8.612595, -8.610714, -8.619876, -8.61291]
# origin of latitude of the flow
x_o = [
    41.148009, 41.140377, 41.149431, 41.146128, 41.145786, 41.140764, 41.150385, 41.150376, 41.148009, 41.150367,
    41.144472, 41.14449, 41.148036, 41.149953, 41.142888, 41.14638, 41.146056, 41.149404, 41.147271, 41.147298,
    41.147298, 41.147388, 41.145957, 41.145669, 41.151294, 41.1462, 41.144634, 41.146164, 41.141313, 41.147937, 41.148,
    41.144544, 41.145615, 41.140485, 41.141187, 41.154354, 41.144544, 41.145606, 41.154525, 41.154372, 41.140746,
    41.145786, 41.145624, 41.151339, 41.145651, 41.145669, 41.150808, 41.144625, 41.148738, 41.145687, 41.142744,
    41.147991, 41.147955, 41.150781, 41.147091, 41.145732, 41.157027, 41.14125, 41.150592, 41.154372, 41.145741,
    41.147928, 41.14575, 41.152806, 41.147379, 41.151312, 41.145696, 41.149917, 41.15007, 41.150034, 41.154417,
    41.150007, 41.141745, 41.145543, 41.145579, 41.144481, 41.145705, 41.145012, 41.145444, 41.156379, 41.146758,
    41.150295, 41.145633, 41.151159, 41.14881, 41.141088, 41.14773, 41.146128, 41.14431, 41.151186, 41.144688,
    41.145498, 41.148, 41.144526, 41.145948, 41.148, 41.145561, 41.15124, 41.141664, 41.145813, 41.145489, 41.145669,
    41.148531, 41.141061, 41.147739, 41.141277, 41.149467, 41.151258, 41.151276, 41.1507, 41.151249, 41.148153,
    41.147415, 41.145732, 41.142798, 41.153598, 41.155569, 41.147163, 41.148063, 41.147991, 41.154399, 41.148306,
    41.143113, 41.147973, 41.140638, 41.149017, 41.145705, 41.145525, 41.147991, 41.150241, 41.150259, 41.140611,
    41.148027, 41.148054, 41.144481, 41.141214, 41.149359, 41.145615, 41.145975, 41.141187, 41.150196, 41.148036,
    41.148063, 41.149278, 41.146146, 41.147847, 41.14089, 41.148513, 41.145534, 41.145786, 41.151429, 41.145588,
    41.14071, 41.141322, 41.146164, 41.148873, 41.146137, 41.149701, 41.151744, 41.146092, 41.155308, 41.150232,
    41.147802, 41.149539, 41.145111, 41.145633, 41.141277, 41.150385, 41.157072, 41.154021, 41.155308, 41.146182,
    41.148018, 41.154417, 41.154426, 41.14557, 41.150259, 41.148, 41.141268, 41.145642, 41.148, 41.14566, 41.153004,
    41.147091, 41.142609, 41.140665, 41.155308, 41.141259, 41.143932, 41.145345, 41.140899, 41.146632, 41.156109,
    41.15556, 41.144634, 41.145993, 41.146146, 41.153499, 41.145768, 41.148045, 41.14485, 41.148837, 41.146074,
    41.147316, 41.145885, 41.1552, 41.145714, 41.140602, 41.146056, 41.14611, 41.140881, 41.140773, 41.148909,
    41.147703, 41.146236, 41.140665, 41.150547, 41.148018, 41.144697, 41.147487, 41.144418, 41.144625, 41.140746,
    41.144715, 41.140674, 41.140764, 41.145588, 41.148045, 41.151051, 41.15043, 41.145795, 41.145795, 41.147973,
    41.147973, 41.144598, 41.142366, 41.148018, 41.155488, 41.150187, 41.15664, 41.149206, 41.153535, 41.146641,
    41.148045, 41.145723, 41.152608, 41.145696, 41.144643, 41.146308, 41.145624, 41.157027, 41.147991, 41.140935,
    41.151438, 41.147865, 41.153472, 41.147676, 41.144544, 41.144715, 41.140728, 41.140413, 41.145885, 41.144625,
    41.149323, 41.148081, 41.144364, 41.14422, 41.144427, 41.144373, 41.145804, 41.148081, 41.144634, 41.146155,
    41.145489, 41.15358, 41.148234, 41.146002, 41.140656, 41.140683, 41.145786, 41.144634, 41.141457, 41.146047,
    41.148981, 41.144616, 41.149206, 41.140683, 41.140368, 41.146065, 41.140728, 41.140287, 41.140683, 41.157036,
    41.156946, 41.145741, 41.140575, 41.140404, 41.144589, 41.140692, 41.140701, 41.15295, 41.140413, 41.142717,
    41.140449, 41.144409, 41.140593, 41.140503, 41.148324, 41.140449, 41.145651, 41.140242, 41.141259, 41.140341,
    41.14071, 41.140548, 41.141349, 41.14035, 41.1471, 41.142357, 41.140377, 41.146812, 41.142168, 41.148, 41.140377,
    41.143617, 41.140404, 41.146056, 41.140368, 41.144517, 41.142213, 41.140494, 41.147793, 41.140359, 41.14926,
    41.153454, 41.146182, 41.144571, 41.146011, 41.140359, 41.146803, 41.148576, 41.140368, 41.140701, 41.140872,
    41.140836, 41.145534, 41.148117, 41.148234, 41.145984, 41.147982, 41.140449]
# destination of latitude of the flow
x_d = [
    41.157351, 41.15592, 41.146839, 41.145489, 41.145786, 41.147019, 41.147325, 41.144094, 41.140944, 41.154732,
    41.140341, 41.144706, 41.154768, 41.152716, 41.153589, 41.14638, 41.142645, 41.140449, 41.147298, 41.147298,
    41.147298, 41.152698, 41.155821, 41.143338, 41.155983, 41.1444, 41.144634, 41.144364, 41.151987, 41.148369,
    41.149881, 41.147784, 41.144454, 41.147892, 41.145615, 41.143302, 41.155191, 41.147649, 41.146875, 41.14179,
    41.145345, 41.148306, 41.147568, 41.147901, 41.152266, 41.140962, 41.148036, 41.147163, 41.149152, 41.147325,
    41.152455, 41.155092, 41.140773, 41.143041, 41.14764, 41.14764, 41.145696, 41.14809, 41.156352, 41.144706,
    41.148567, 41.143986, 41.144463, 41.144103, 41.144373, 41.14728, 41.145786, 41.150079, 41.15007, 41.150034,
    41.151627, 41.150007, 41.154318, 41.150592, 41.155155, 41.147595, 41.142528, 41.146794, 41.155218, 41.153535,
    41.14674, 41.150313, 41.141853, 41.15115, 41.152122, 41.143005, 41.144247, 41.149962, 41.14431, 41.155263,
    41.148126, 41.155056, 41.151663, 41.14755, 41.15088, 41.145642, 41.149908, 41.146128, 41.149773, 41.155344,
    41.154984, 41.153517, 41.147847, 41.147262, 41.147622, 41.148774, 41.147667, 41.144472, 41.15754, 41.148, 41.144094,
    41.151555, 41.145588, 41.144454, 41.15106, 41.156388, 41.148369, 41.148738, 41.14422, 41.14431, 41.148324,
    41.148306, 41.149593, 41.144508, 41.147937, 41.148936, 41.145066, 41.145525, 41.14899, 41.140566, 41.150259,
    41.148999, 41.153409, 41.146335, 41.156604, 41.148756, 41.156487, 41.149755, 41.141547, 41.151546, 41.146776,
    41.145237, 41.142861, 41.14935, 41.144481, 41.150709, 41.152356, 41.147055, 41.144382, 41.141439, 41.148027,
    41.145588, 41.156694, 41.14521, 41.148189, 41.147604, 41.152617, 41.145552, 41.140377, 41.140755, 41.144121,
    41.153058, 41.155317, 41.147604, 41.140575, 41.149989, 41.148261, 41.14836, 41.150628, 41.145615, 41.153643,
    41.144301, 41.151654, 41.149143, 41.147604, 41.149674, 41.140809, 41.144463, 41.157135, 41.144409, 41.145849,
    41.152365, 41.144481, 41.141394, 41.14134, 41.146803, 41.145417, 41.149746, 41.148675, 41.141646, 41.147253,
    41.150646, 41.142159, 41.146704, 41.149854, 41.150187, 41.150565, 41.146533, 41.14989, 41.15448, 41.157063,
    41.149125, 41.146074, 41.148459, 41.140458, 41.155173, 41.151897, 41.151051, 41.144058, 41.142564, 41.146092,
    41.145921, 41.148891, 41.147964, 41.15088, 41.146884, 41.148828, 41.145354, 41.150664, 41.153571, 41.144418,
    41.144652, 41.148855, 41.14305, 41.146884, 41.140764, 41.152374, 41.151087, 41.14602, 41.15043, 41.145867,
    41.148252, 41.155443, 41.150529, 41.143599, 41.150502, 41.147757, 41.15331, 41.145723, 41.148207, 41.149206,
    41.15493, 41.146317, 41.144481, 41.147676, 41.144355, 41.151537, 41.152212, 41.156352, 41.146011, 41.14872,
    41.148342, 41.148369, 41.151438, 41.147514, 41.14782, 41.154066, 41.146785, 41.141727, 41.145732, 41.149593,
    41.146713, 41.144202, 41.143554, 41.15457, 41.144229, 41.144418, 41.144373, 41.144481, 41.139216, 41.148252,
    41.147424, 41.144472, 41.147541, 41.155443, 41.140638, 41.140413, 41.151969, 41.150574, 41.15331, 41.149143,
    41.15007, 41.150178, 41.148981, 41.147577, 41.149143, 41.146686, 41.151744, 41.154768, 41.140566, 41.150232,
    41.147847, 41.145066, 41.154552, 41.140755, 41.152194, 41.149953, 41.146794, 41.147775, 41.143275, 41.148531,
    41.147028, 41.148927, 41.146002, 41.146875, 41.1498, 41.150259, 41.148324, 41.150088, 41.1453, 41.150151, 41.150034,
    41.147172, 41.152752, 41.149908, 41.150061, 41.151168, 41.157207, 41.155191, 41.150187, 41.141709, 41.14557,
    41.151735, 41.145912, 41.143617, 41.145318, 41.154678, 41.150088, 41.143581, 41.154966, 41.146893, 41.153139,
    41.146551, 41.140548, 41.148198, 41.144139, 41.144112, 41.142096, 41.143167, 41.157234, 41.146173, 41.154867,
    41.147982, 41.147766, 41.147766, 41.153895, 41.155983, 41.150403, 41.154921, 41.143491, 41.150907]
# destination of longitude of the flow
y_d = [
    -8.60949, -8.60247, -8.619993, -8.607465, -8.610912, -8.620146, -8.602524, -8.599419, -8.615232, -8.604756,
    -8.613072, -8.606322, -8.618823, -8.611443, -8.605575, -8.617662, -8.615358, -8.613063, -8.615808, -8.615817,
    -8.615835, -8.613684, -8.609229, -8.602704, -8.611866, -8.605836, -8.606484, -8.60607, -8.613765, -8.613855,
    -8.604999, -8.62011, -8.606052, -8.607753, -8.610795, -8.618094, -8.604981, -8.62011, -8.620101, -8.616537,
    -8.60589, -8.613585, -8.620146, -8.619102, -8.618769, -8.614683, -8.606439, -8.617788, -8.609715, -8.617041,
    -8.607321, -8.612712, -8.615196, -8.614746, -8.608698, -8.622171, -8.610822, -8.61129, -8.620479, -8.60679,
    -8.611992, -8.607321, -8.60607, -8.604783, -8.605683, -8.61777, -8.610723, -8.620992, -8.621001, -8.620992,
    -8.609517, -8.620983, -8.609058, -8.608959, -8.612406, -8.622144, -8.61183, -8.620065, -8.60607, -8.606718,
    -8.620002, -8.602254, -8.616438, -8.609454, -8.61804, -8.608311, -8.606124, -8.619129, -8.605458, -8.607681,
    -8.618157, -8.613126, -8.609508, -8.60931, -8.602731, -8.607078, -8.619561, -8.607096, -8.61021, -8.606637,
    -8.613144, -8.607069, -8.614602, -8.617365, -8.621946, -8.610804, -8.622063, -8.606097, -8.604675, -8.616033,
    -8.607546, -8.609445, -8.610696, -8.605881, -8.600724, -8.6202, -8.607987, -8.60751, -8.605251, -8.622288,
    -8.608005, -8.614296, -8.620848, -8.606034, -8.610903, -8.612019, -8.605764, -8.610867, -8.607933, -8.615772,
    -8.607024, -8.607735, -8.607222, -8.611254, -8.621397, -8.607834, -8.620308, -8.619912, -8.613558, -8.609355,
    -8.619948, -8.610705, -8.614746, -8.603406, -8.606259, -8.606925, -8.606997, -8.608887, -8.606376, -8.61453,
    -8.607654, -8.611011, -8.621721, -8.610696, -8.60814, -8.621919, -8.615817, -8.617527, -8.613036, -8.610327,
    -8.607456, -8.618157, -8.604792, -8.620092, -8.615799, -8.619183, -8.613189, -8.614179, -8.606925, -8.607231,
    -8.613162, -8.606124, -8.609535, -8.612982, -8.620164, -8.620812, -8.615205, -8.605989, -8.602047, -8.606007,
    -8.61084, -8.618976, -8.606097, -8.615241, -8.607816, -8.620047, -8.615592, -8.62101, -8.602479, -8.61633,
    -8.618022, -8.610912, -8.608509, -8.611173, -8.61957, -8.613054, -8.600256, -8.617707, -8.614008, -8.621001,
    -8.613018, -8.609517, -8.61903, -8.60643, -8.613108, -8.607735, -8.606367, -8.604333, -8.605107, -8.603127,
    -8.61066, -8.61318, -8.611812, -8.610903, -8.60679, -8.620119, -8.616213, -8.607231, -8.610165, -8.608914,
    -8.606448, -8.606502, -8.6121, -8.60238, -8.60247, -8.609679, -8.620785, -8.602578, -8.611398, -8.6067, -8.59968,
    -8.606277, -8.614431, -8.610975, -8.617329, -8.602263, -8.620101, -8.604855, -8.610804, -8.613216, -8.611164,
    -8.61894, -8.617869, -8.605917, -8.602785, -8.606115, -8.617698, -8.604846, -8.611821, -8.612982, -8.607186,
    -8.608122, -8.606115, -8.617644, -8.601867, -8.6094, -8.611983, -8.60193, -8.612937, -8.614935, -8.604639,
    -8.620353, -8.599437, -8.602776, -8.614404, -8.609094, -8.609067, -8.60904, -8.609004, -8.609211, -8.608077,
    -8.620551, -8.60481, -8.620227, -8.618751, -8.611344, -8.613036, -8.606133, -8.621082, -8.60346, -8.603019,
    -8.62092, -8.620947, -8.620695, -8.611335, -8.599257, -8.605215, -8.609661, -8.620569, -8.616159, -8.620938,
    -8.606511, -8.604423, -8.602416, -8.610714, -8.606988, -8.620974, -8.619966, -8.606502, -8.607474, -8.616087,
    -8.606907, -8.613036, -8.617653, -8.620092, -8.620974, -8.620803, -8.612757, -8.620947, -8.610696, -8.621001,
    -8.621001, -8.616267, -8.604855, -8.62092, -8.620839, -8.614809, -8.612748, -8.602812, -8.620938, -8.618454,
    -8.607402, -8.60967, -8.606898, -8.617311, -8.607312, -8.616249, -8.620857, -8.607843, -8.613297, -8.602497,
    -8.602641, -8.614584, -8.612973, -8.615727, -8.599518, -8.613423, -8.61453, -8.617536, -8.601903, -8.619606,
    -8.610066, -8.612631, -8.609949, -8.613972, -8.613387, -8.617644, -8.614701, -8.604594, -8.612046, -8.604693]


class Net:
    def __init__(self, node2coord, edge):
        self.edge = edge
        self.node2coord = node2coord
        self.xs = []
        self.ys = []
        self.intersections = self.getIntersections()
        self.graph = self.generate_graph()  # undigraph
        self.roads = self.getRoads()
        self.laneLink_point = []
        self.lane_num = 3
        self.add_roadlinks(self.intersections)
        self.flows = self.add_flows()

    def generate_graph(self):
        G = nx.Graph()
        for x, y in zip(self.xs, self.ys):
            G.add_node((x, y))

        weighted_edges = []
        for e in self.edge:
            p1 = self.intersections[e[0]].point
            p2 = self.intersections[e[1]].point
            weight = p1.distance(p2)
            ebunch = list(e)
            ebunch.append(weight)  # 3-d tuple: (o, d, v)]
            weighted_edges.append(tuple(ebunch))

        G.add_weighted_edges_from(weighted_edges)

        return G

    def getIntersections(self):
        # points = []
        xs = []
        ys = []
        # inter_x = []
        # inter_y = []
        intersections = []
        for i, v in enumerate(self.node2coord.values()):
            x, y = self.MercatorToxy(v[0], v[1])
            xs.append(x)
            ys.append(y)
        self.xs = xs
        self.ys = ys
        axis_x = min(xs)
        axis_y = min(ys)
        for i, _ in enumerate(xs):
            point = Point(xs[i] - axis_x, ys[i] - axis_y, evaluate=False)
            inter = Intersection(i, point)
            intersections.append(inter)
            # print("%s is created! " % inter.id_inter)
            # print(inter.x, inter.y)
            # inter_x.append(inter.x)
            # inter_y.append(inter.y)

        self.add_roads(self.edge, intersections)
        # self.add_trafficLight(intersections)
        print("---------- there is %d intersections in the net---------" % len(intersections))
        # print(inter_x)
        # print(inter_y)

        return intersections

    def getRoads(self):
        roads = []
        # for i, e in enumerate(self.edge):
        #     # print("edge index0:", e[0], "\n")
        #     r1 = Road(self.intersections[e[0]], self.intersections[e[1]])
        #     roads.append(r1)
        #     print("%s is created!" % r1.id_road)
        #     r2 = Road(self.intersections[e[1]], self.intersections[e[0]])
        #     roads.append(r2)
        #     print("%s created!" % r2.id_road)
        for inter in self.intersections:
            inter.Enterroads = self.sort_enterroads_by_clock(inter.Enterroads)
            for road in inter.Enterroads:
                roads.append(road)
                print("%s is created!" % road.id_road)
        # print("---------- there is %d roads in the net---------" % len(roads))
        return roads

    def add_roads(self, edge, intersections):
        roads = []
        inter_type = [0, 0, 0, 0, 0, 0]
        for i, e in enumerate(edge):
            R1 = Road(intersections[e[0]], intersections[e[1]])
            R2 = Road(intersections[e[1]], intersections[e[0]])
            intersections[e[0]].Exitroads.append(R1)
            intersections[e[1]].Enterroads.append(R1)
            intersections[e[0]].Enterroads.append(R2)
            intersections[e[1]].Exitroads.append(R2)

        for inter in intersections:
            inter.Enterroads = self.sort_enterroads_by_clock(inter.Enterroads)
            # for road in inter.Enterroads:
            #     roads.append(road)
            #     print("%s is created!" % road.id_road)
            inter.roads = inter.Exitroads + inter.Enterroads
            inter.type = len(inter.Exitroads)
            # inter.trafficLight = inter.add_trafficLight()
            inter_type[inter.type - 2] = inter_type[inter.type - 2] + 1

        print("-------------intersection_roads built!-----------------\n", inter_type)
        print("---------- there is %d roads in the net---------" % len(roads))

    def add_roadlinks(self, intersections):
        for i, inter in enumerate(intersections):
            roadLinks = []
            for enterroad in intersections[i].Enterroads:
                for exitroad in intersections[i].Exitroads:
                    if exitroad.end_point == enterroad.start_point:
                        continue
                    t = self.get_roadlink_type(enterroad, exitroad)
                    type = roadLink_type[t]
                    laneLinks = self.add_laneLinks(enterroad, exitroad, t)
                    roadLink = {
                        # 'turn_left', 'turn_right', 'go_straight'
                        "type": type,
                        # id of starting road
                        "startRoad": enterroad.id_road,
                        # id of ending road
                        "endRoad": exitroad.id_road,
                        # lanelinks of roadlink
                        "laneLinks": laneLinks,
                    }

                    roadLinks.append(roadLink)
                    inter.roadLinks = roadLinks
                    inter.trafficLight = inter.add_trafficLight()
                    # print(len(inter.roadLinks))

    def get_roadlink_type(self, enterroad, exitroad):  # TODO
        # print("get roadlink type of %s%s" % (enterroad.id_road,exitroad.id_road))
        angle = self.get_re_angel(enterroad.start_point, exitroad.start_point, exitroad.end_point)
        if (angle < math.pi / 3 or angle > math.pi / 3 * 5):
            direction = 1  # go straight
            # type = "go_straight"
        elif (angle > math.pi):
            direction = 2  # turn right
            # type = "turn_right"
        else:
            direction = 0  # turn left
            # type = "turn_left"
        return direction

    def add_laneLinks(self, enterroad, exitroad, t):
        # print("add_laneLinks...")
        lanelinks = []
        for s_lane in enterroad.lanes:
            if t == s_lane.index:
                for e_lane in exitroad.lanes:  # 3 choices depending on the route
                    lanelink = self.get_laneLink(s_lane, e_lane)
                    lanelinks.append(lanelink)
        # print("%d laneLinks" % t, lanelinks)
        return lanelinks

    def sort_enterroads_by_clock(self, enterroads):
        angles = []  # ab_angle = atan2(enterroad.endpoint - enterroad.startpoint)
        for road in enterroads:
            angles.append(road.angle)
        sort_road_index = np.argsort(angles)
        sort_enterroads = []
        for i in sort_road_index:
            sort_enterroads.append(enterroads[i])
        return sort_enterroads

    def get_laneLink(self, s_lane, e_lane):
        start_point = s_lane.end_point
        # print(type(start_point), "!!!!!!!!!!!!!!!!!!!!!!!!!!")
        end_point = e_lane.start_point
        start_point = {
            "x": float(start_point.x),
            "y": float(start_point.y)
        }
        end_point = {
            "x": float(end_point.x),
            "y": float(end_point.y)
        }
        points = [start_point, end_point]
        lanelink = {
            # from startRoad's startLaneIndex lane to endRoad's endLaneIndex lane
            "startLaneIndex": s_lane.index,
            "endLaneIndex": e_lane.index,
            # points along the laneLink which describe the shape of laneLink
            "points": points,
        }
        return lanelink

    def add_lightphases(self):
        pass

    def is_inter_virtual(self):
        pass

    def millerTOxy(self, lon, lat):
        # TODO: Miller projection method is used to convert the earth's latitude and longitude coordinates to Cartesian coordinates
        L = 6381372 * math.pi * 2  # arth circumference
        W = L  # expand the plane, think of the perimeter as the X-axis
        H = L / 2  # The Y-axis is about half the circumference
        mill = 2.3  # A constant in miller's projection
        x = lon * math.pi / 180  # Convert longitude from degrees to radians
        y = lat * math.pi / 180

        y = 1.25 * math.log(math.tan(0.25 * math.pi + 0.4 * y))  # transformation of miller's projection

        # convert radians to actual distances in kilometers
        x = (W / 2) + (W / (2 * math.pi)) * x * 1000
        y = (H / 2) - (H / (2 * mill)) * y * 1000  #
        return x, y

    def MercatorToxy(self, lon, lat):
        x = lon * 20037508.342789 / 180
        y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
        y = y * 20037508.34789 / 180
        return x, y

    def get_re_angel(self, p1, p2, p3):
        x1 = p2.x - p1.x
        x2 = p3.x - p2.x
        y1 = p2.y - p1.y
        y2 = p3.y - p2.y
        dot = x1 * x2 + y1 * y2
        det = x1 * y2 - y1 * x2
        theta = math.atan2(det, dot)
        theta = theta if theta > 0 else 2 * math.pi + theta
        return theta

    def get_ab_angle(self, pre_point, now_point):  # tell the direction from inter A to B (edge:(A,B)
        pre_angle = math.atan2(pre_point.y, pre_point.x) * 180 / math.pi
        now_angle = math.atan2(now_point.y, now_point.x) * 180 / math.pi
        angle_diff = now_angle - pre_angle
        # if (angle_diff < 0):  # 保证角度>0
        #     angle_diff += math.pi * 2
        return angle_diff

    def get_road_direction(self, start_point, end_point):
        angle = self.get_clockwise_angle(start_point, end_point)
        if (angle >= 7 * math.pi / 4 and angle < math.pi / 4):
            return 0  # right
        elif (angle >= math.pi / 4 and angle < math.pi / 4 * 3):
            return 1  # up
        elif (angle < 7 * math.pi / 4 and angle >= 5 * math.pi / 4):
            return 3  # down
        else:
            return 2  # left

    def get_clockwise_angle(self, p1, p2):
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        dot = x1 * x2 + y1 * y2
        det = x1 * y2 - y1 * x2
        theta = math.atan2(det, dot)
        theta = theta if theta > 0 else 2 * np.pi + theta
        return theta

    def add_flows(self):
        flows = []
        for i in range(len(x_o)):
            f = Flow(self.node2coord, i, x_o[i], y_o[i], x_d[i], y_d[i])
            f.start_inter = self.intersections[f.origin_inter_iid]
            f.end_inter = self.intersections[f.des_inter_iid]
            f.route = self.get_flow_route(f)
            flows.append(f)

        return flows

    def get_flow_route(self, flow):
        route = []
        inter_index = []
        s = flow.origin_inter_iid
        t = flow.des_inter_iid
        # print("-------------flow %d---------------" % flow.iid)

        # return []
        inter_route = list(nx.all_shortest_paths(self.graph, source=s, target=t))
        if len(inter_route[0]) != 0:
            inter_index = inter_route[0]
            # print("flow %d inter_route index is:" % flow.iid, inter_route)
        else:
            print("No route from intersection_%d to intersection_%d!\n" % (s, t))

        for i in range(len(inter_index) - 1):
            for exit_road in self.intersections[inter_index[i]].Exitroads:
                if exit_road.endIntersection.iid == inter_index[i + 1]:
                    route.append(exit_road.id_road)
                    break
        if len(route) != len(inter_index) - 1:
            pass
            # print('Error happened when loading route: %s roads is missed\n' % (len(inter_index) -1 - len(route)))
        # print(route)
        return route

    def get_optional_routes(self, flow):
        LRR_inter_index = []
        inter_index_3 = []
        inter_index_2 = []
        s = flow.origin_inter_iid
        t = flow.des_inter_iid

        # Get shortest route
        inter_route_shortest = nx.dijkstra_path(self.graph, source=s, target=t)
        if len(inter_route_shortest) != 0:
            pass
            # print("flow %d inter_route index is:" % flow.iid, inter_route_shortest)
        else:
            print("No shortest route from intersection_%d to intersection_%d!\n" % (s, t))

        print("index", inter_route_shortest)
        route_1 = self.get_route_from_index(inter_route_shortest)
        print("rou1:", route_1)
        print(len(inter_route_shortest))
        if len(inter_route_shortest) == 1 or len(inter_route_shortest) == 2:
            print("-----len are 1 or 2------")
            return route_1, route_1, route_1
        # Get route containing the least roads(LRR)
        inter_route_least = list(nx.all_shortest_paths(self.graph, source=s, target=t))
        print(flow.iid, "least roads route:", inter_route_least)

        start1 = inter_route_shortest[1]
        start2 = inter_route_shortest[2]

        if len(inter_route_least[0]) != 0:
            if inter_route_least[0][1] == start1 and inter_route_least[0][2] == start2:
                if inter_route_shortest == inter_route_least[0]:
                    print("route_1 and route_2 repeated")
                    # check if there is another "LRR"
                    try:
                        if inter_route_least[1] == None:
                            pass
                    except(IndexError):
                        print("route_2 has no alternative route!")

                        # take 2 simple road instead
                        for path in nx.all_simple_paths(self.graph, source=s, target=t):
                            # the third route cant repeat, search until get a new route
                            if (path[1] == start1 and path[
                                2] == start2 and path != inter_route_least and path != LRR_inter_index):
                                inter_index_2 = path
                                break
                        if inter_index_2 == []:
                            print("flow %d route_2 missed!" % flow.iid)

                        for path in nx.all_simple_paths(self.graph, source=s, target=t):
                            # the third route cant repeat, search until get a new route
                            if (path[1] == start1 and path[
                                2] == start2 and path != inter_route_least and path != LRR_inter_index):
                                inter_index_3 = path
                                break
                        if inter_index_3 == []:
                            print("flow %d route_3 missed!" % flow.iid)


                    # has 2 routes already, one needed
                    else:
                        LRR_inter_index = inter_route_least[1]
                        for path in nx.all_simple_paths(self.graph, source=s, target=t):
                            # the third route cant repeat, search until get a new route
                            if (path != inter_route_least and path != LRR_inter_index and
                                    path[1] == start1):
                                inter_index_3 = path
                                break
                        if inter_index_3 == []:
                            print("flow %d route_3 missed!" % flow.iid)

                else:
                    LRR_inter_index = inter_route_least[0]
            # print("flow %d inter_route index is:" % flow.iid, inter_route_least)
            else:
                print("No SAME2START shortest route from intersection_%d to intersection_%d!\n" % (s, t))
                for path in nx.all_simple_paths(self.graph, source=s, target=t):
                    # the third route cant repeat, search until get a new route
                    if (path[1] == start1 and path[
                        2] == start2 and path != inter_route_least and path != LRR_inter_index):
                        inter_index_2 = path
                        break
                if inter_index_2 == []:
                    print("flow %d route_2 missed!" % flow.iid)

                for path in nx.all_simple_paths(self.graph, source=s, target=t):
                    # the third route cant repeat, search until get a new route
                    if (path[1] == start1 and path[
                        2] == start2 and path != inter_route_least and path != LRR_inter_index):
                        inter_index_3 = path
                        break
                if inter_index_3 == []:
                    print("flow %d route_3 missed!" % flow.iid)
        else:
            print("No extra shortest route from intersection_%d to intersection_%d!\n" % (s, t))

        if (LRR_inter_index != []):
            route_2 = self.get_route_from_index(LRR_inter_index)
        else:
            route_2 = self.get_route_from_index(inter_index_2)
        route_3 = self.get_route_from_index(inter_index_3)

        return route_1, route_2, route_3

        # Get

    def get_route_from_index(self, index):
        route = []
        for i in range(len(index) - 1):
            for exit_road in self.intersections[index[i]].Exitroads:
                if exit_road.endIntersection.iid == index[i + 1]:
                    route.append(exit_road.id_road)
                    break
        if len(route) != len(index) - 1:
            print('Error happened when loading route: %s roads is missed\n' % (len(index) - 1 - len(route)))
        return route
