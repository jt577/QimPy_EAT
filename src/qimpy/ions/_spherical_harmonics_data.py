# Clebsch-Gordon coefficients for products of real harmonics.
# The integer indices correspond to l*(l+1)+m for each (l,m).
YLM_PROD = {
    (0, 0): ([0], [0.2820947917738784]),
    (1, 0): ([1], [0.2820947917738780]),
    (1, 1): ([0, 6, 8],
             [0.2820947917738786, -0.1261566261010079, -0.2185096861184159]),
    (2, 0): ([2], [0.2820947917738775]),
    (2, 1): ([5], [0.2185096861184158]),
    (2, 2): ([0, 6], [0.2820947917738781, 0.2523132522020159]),
    (3, 0): ([3], [0.2820947917738785]),
    (3, 1): ([4], [0.2185096861184158]),
    (3, 2): ([7], [0.2185096861184158]),
    (3, 3): ([0, 6, 8],
             [0.2820947917738783, -0.1261566261010080, 0.2185096861184160]),
    (4, 0): ([4], [0.2820947917738783]),
    (4, 1): ([3, 13, 15],
             [0.2185096861184159, -0.0583991700819018, -0.2261790131595404]),
    (4, 2): ([10], [0.1846743909223718]),
    (4, 3): ([1, 9, 11],
             [0.2185096861184158, 0.2261790131595401, -0.0583991700819020]),
    (4, 4): ([0, 6, 20, 24],
             [0.2820947917738783, -0.1802237515728686, 0.0402992559676969,
              -0.2384136135044480]),
    (5, 0): ([5], [0.2820947917738781]),
    (5, 1): ([2, 12, 14],
             [0.2185096861184159, -0.1430481681026690, -0.1846743909223719]),
    (5, 2): ([1, 11], [0.2185096861184158, 0.2335966803276075]),
    (5, 3): ([10], [0.1846743909223718]),
    (5, 4): ([7, 21, 23],
             [0.1560783472274398, -0.0637187184340276, -0.1685838828361838]),
    (5, 5): ([0, 6, 8, 20, 22],
             [0.2820947917738783, 0.0901118757864345, -0.1560783472274400,
              -0.1611970238707874, -0.1802237515728688]),
    (6, 0): ([6], [0.2820947917738780]),
    (6, 1): ([1, 11], [-0.1261566261010080, 0.2023006594034207]),
    (6, 2): ([2, 12], [0.2523132522020162, 0.2477666950834761]),
    (6, 3): ([3, 13], [-0.1261566261010080, 0.2023006594034206]),
    (6, 4): ([4, 18], [-0.1802237515728686, 0.1560783472274399]),
    (6, 5): ([5, 19], [0.0901118757864342, 0.2207281154418226]),
    (6, 6): ([0, 6, 20],
             [0.2820947917738781, 0.1802237515728686, 0.2417955358061813]),
    (7, 0): ([7], [0.2820947917738779]),
    (7, 1): ([10], [0.1846743909223718]),
    (7, 2): ([3, 13], [0.2185096861184160, 0.2335966803276075]),
    (7, 3): ([2, 12, 14],
             [0.2185096861184160, -0.1430481681026689, 0.1846743909223719]),
    (7, 4): ([5, 17, 19],
             [0.1560783472274400, 0.1685838828361838, -0.0637187184340276]),
    (7, 5): ([4, 18], [0.1560783472274399, 0.1802237515728689]),
    (7, 6): ([7, 21], [0.0901118757864343, 0.2207281154418227]),
    (7, 7): ([0, 6, 8, 20, 22],
             [0.2820947917738783, 0.0901118757864343, 0.1560783472274400,
              -0.1611970238707875, 0.1802237515728687]),
    (8, 0): ([8], [0.2820947917738783]),
    (8, 1): ([1, 9, 11],
             [-0.2185096861184157, 0.2261790131595401, 0.0583991700819019]),
    (8, 2): ([14], [0.1846743909223718]),
    (8, 3): ([3, 13, 15],
             [0.2185096861184159, -0.0583991700819020, 0.2261790131595405]),
    (8, 4): ([16], [0.2384136135044484]),
    (8, 5): ([5, 17, 19],
             [-0.1560783472274400, 0.1685838828361839, 0.0637187184340276]),
    (8, 6): ([8, 22], [-0.1802237515728686, 0.1560783472274399]),
    (8, 7): ([7, 21, 23],
             [0.1560783472274398, -0.0637187184340276, 0.1685838828361839]),
    (8, 8): ([0, 6, 20, 24],
             [0.2820947917738782, -0.1802237515728687, 0.0402992559676969,
              0.2384136135044481]),
    (9, 0): ([9], [0.2820947917738781]),
    (9, 1): ([8, 22, 24],
             [0.2261790131595403, -0.0435281713775681, -0.2303294329808905]),
    (9, 2): ([17], [0.1628675039676399]),
    (9, 3): ([4, 16, 18],
             [0.2261790131595405, 0.2303294329808903, -0.0435281713775682]),
    (9, 4): ([3, 13, 31, 35],
             [0.2261790131595403, -0.0940315972579596, 0.0169433177293594,
              -0.2455320005465368]),
    (9, 5): ([14, 32, 34],
             [0.1486770096793976, -0.0448278050962362, -0.1552880720369527]),
    (9, 6): ([9, 27], [-0.2102610435016800, 0.1267921798770303]),
    (9, 7): ([10, 26, 28],
             [0.1486770096793975, 0.1552880720369528, -0.0448278050962363]),
    (9, 8): ([1, 11, 25, 29],
             [0.2261790131595404, -0.0940315972579596, 0.2455320005465369,
              0.0169433177293594]),
    (9, 9): ([0, 6, 20, 42, 48],
             [0.2820947917738780, -0.2102610435016800, 0.0769349432110579,
              -0.0118543966932640, -0.2548005986729749]),
    (10, 0): ([10], [0.2820947917738783]),
    (10, 1): ([7, 21, 23],
              [0.1846743909223716, -0.0753930043865135, -0.1994711402007163]),
    (10, 2): ([4, 18], [0.1846743909223719, 0.2132436186229229]),
    (10, 3): ([5, 17, 19],
              [0.1846743909223720, 0.1994711402007162, -0.0753930043865135]),
    (10, 4): ([2, 12, 30, 34],
              [0.1846743909223717, -0.1880631945159189, 0.0535794751446877,
               -0.1901882698155454]),
    (10, 5): ([3, 13, 15, 31, 33],
              [0.1846743909223718, 0.1151647164904451, -0.1486770096793977,
               -0.0830049659735642, -0.1793112203849455]),
    (10, 6): ([28], [0.1901882698155454]),
    (10, 7): ([1, 9, 11, 27, 29],
              [0.1846743909223726, 0.1486770096793975, 0.1151647164904452,
               0.1793112203849457, -0.0830049659735642]),
    (10, 8): ([26], [0.1901882698155457]),
    (10, 9): ([7, 21, 43, 47],
              [0.1486770096793975, -0.0993225845992799, 0.0221775454765501,
               -0.1801712311720527]),
    (10, 10): ([0, 20, 24, 42, 46],
               [0.2820947917738785, -0.1795148674924678, -0.1517177540482851,
                0.0711263801595843, -0.1881827135584984]),
    (11, 0): ([11], [0.2820947917738781]),
    (11, 1): ([6, 8, 20, 22],
              [0.2023006594034207, 0.0583991700819020, -0.1507860087730268,
               -0.1685838828361840]),
    (11, 2): ([5, 19], [0.2335966803276073, 0.2384136135044479]),
    (11, 3): ([4, 18], [-0.0583991700819018, 0.1685838828361840]),
    (11, 4): ([3, 13, 15, 31, 33],
              [-0.0583991700819021, 0.1456731240789440, 0.0940315972579593,
               -0.0656211873953096, -0.1417579666102106]),
    (11, 5): ([2, 12, 14, 30, 32],
              [0.2335966803276075, 0.0594708038717588, -0.1151647164904453,
               -0.1694331772935930, -0.1736173425847552]),
    (11, 6): ([1, 11, 29],
              [0.2023006594034205, 0.1261566261010081, 0.2273184612433489]),
    (11, 7): ([10, 28], [0.1151647164904452, 0.1736173425847554]),
    (11, 8): ([1, 9, 11, 27, 29],
              [0.0583991700819018, -0.0940315972579593, -0.1456731240789438,
               0.1417579666102103, 0.0656211873953096]),
    (11, 9): ([8, 22, 24, 44, 46],
              [-0.0940315972579595, 0.1332552305189783, 0.1175200669506004,
               -0.0443550909531000, -0.1214714192760309]),
    (11, 10): ([7, 21, 23, 43, 45],
               [0.1151647164904453, 0.1025799242814103, -0.0678502422891120,
                -0.0858932642904355, -0.1629710104947501]),
    (11, 11): ([0, 6, 8, 20, 22, 42, 44],
               [0.2820947917738783, 0.1261566261010081, -0.1456731240789437,
                0.0256449810703525, -0.1146878419100074, -0.1778159503989609,
                -0.1717865285808715]),
    (12, 0): ([12], [0.2820947917738782]),
    (12, 1): ([5, 19], [-0.1430481681026688, 0.1946639002730061]),
    (12, 2): ([6, 20], [0.2477666950834761, 0.2462325212298290]),
    (12, 3): ([7, 21], [-0.1430481681026688, 0.1946639002730063]),
    (12, 4): ([10, 28], [-0.1880631945159187, 0.1417579666102105]),
    (12, 5): ([1, 11, 29],
              [-0.1430481681026687, 0.0594708038717590, 0.2143179005787512]),
    (12, 6): ([2, 12, 30],
              [0.2477666950834761, 0.1682088348013440, 0.2396146972445646]),
    (12, 7): ([3, 13, 31],
              [-0.1430481681026689, 0.0594708038717591, 0.2143179005787512]),
    (12, 8): ([14, 32], [-0.1880631945159188, 0.1417579666102104]),
    (12, 9): ([17, 39], [-0.2035507268673357, 0.1086473403298334]),
    (12, 10): ([4, 18, 40],
               [-0.1880631945159189, -0.0444184101729925, 0.1774203638124001]),
    (12, 11): ([5, 19, 41],
               [0.0594708038717593, 0.0993225845992801, 0.2217754547655002]),
    (12, 12): ([0, 6, 20, 42],
               [0.2820947917738782, 0.1682088348013438, 0.1538698864221151,
                0.2370879338652807]),
    (13, 0): ([13], [0.2820947917738783]),
    (13, 1): ([4, 18], [-0.0583991700819019, 0.1685838828361840]),
    (13, 2): ([7, 21], [0.2335966803276073, 0.2384136135044480]),
    (13, 3): ([6, 8, 20, 22],
              [0.2023006594034207, -0.0583991700819019, -0.1507860087730269,
               0.1685838828361840]),
    (13, 4): ([1, 9, 11, 27, 29],
              [-0.0583991700819019, -0.0940315972579593, 0.1456731240789441,
               0.1417579666102106, -0.0656211873953097]),
    (13, 5): ([10, 28], [0.1151647164904452, 0.1736173425847554]),
    (13, 6): ([3, 13, 31],
              [0.2023006594034206, 0.1261566261010081, 0.2273184612433490]),
    (13, 7): ([2, 12, 14, 30, 32],
              [0.2335966803276073, 0.0594708038717590, 0.1151647164904451,
               -0.1694331772935934, 0.1736173425847552]),
    (13, 8): ([3, 13, 15, 31, 33],
              [-0.0583991700819019, 0.1456731240789441, -0.0940315972579593,
               -0.0656211873953097, 0.1417579666102106]),
    (13, 9): ([4, 16, 18, 38, 40],
              [-0.0940315972579594, -0.1175200669506002, 0.1332552305189780,
               0.1214714192760310, -0.0443550909531002]),
    (13, 10): ([5, 17, 19, 39, 41],
               [0.1151647164904451, 0.0678502422891119, 0.1025799242814098,
                0.1629710104947499, -0.0858932642904357]),
    (13, 11): ([4, 18, 40],
               [0.1456731240789438, 0.1146878419100072, 0.1717865285808715]),
    (13, 12): ([7, 21, 43],
               [0.0594708038717590, 0.0993225845992798, 0.2217754547654997]),
    (13, 13): ([0, 6, 8, 20, 22, 42, 44],
               [0.2820947917738783, 0.1261566261010080, 0.1456731240789440,
                0.0256449810703525, 0.1146878419100072, -0.1778159503989609,
                0.1717865285808716]),
    (14, 0): ([14], [0.2820947917738782]),
    (14, 1): ([5, 17, 19],
              [-0.1846743909223719, 0.1994711402007163, 0.0753930043865134]),
    (14, 2): ([8, 22], [0.1846743909223716, 0.2132436186229232]),
    (14, 3): ([7, 21, 23],
              [0.1846743909223717, -0.0753930043865136, 0.1994711402007163]),
    (14, 4): ([26], [0.1901882698155457]),
    (14, 5): ([1, 9, 11, 27, 29],
              [-0.1846743909223718, 0.1486770096793974, -0.1151647164904451,
               0.1793112203849453, 0.0830049659735640]),
    (14, 6): ([32], [0.1901882698155455]),
    (14, 7): ([3, 13, 15, 31, 33],
              [0.1846743909223718, 0.1151647164904453, 0.1486770096793976,
               -0.0830049659735642, 0.1793112203849456]),
    (14, 8): ([2, 12, 30, 34],
              [0.1846743909223719, -0.1880631945159189, 0.0535794751446878,
               0.1901882698155457]),
    (14, 9): ([5, 19, 37, 41],
              [0.1486770096793981, -0.0993225845992790, 0.1801712311720540,
               0.0221775454765485]),
    (14, 10): ([16, 38], [0.1517177540482849, 0.1881827135584986]),
    (14, 11): ([5, 17, 19, 39, 41],
               [-0.1151647164904450, 0.0678502422891118, -0.1025799242814103,
                0.1629710104947500, 0.0858932642904360]),
    (14, 12): ([8, 22, 44],
               [-0.1880631945159189, -0.0444184101729926, 0.1774203638123999]),
    (14, 13): ([7, 21, 23, 43, 45],
               [0.1151647164904452, 0.1025799242814104, 0.0678502422891119,
                -0.0858932642904359, 0.1629710104947502]),
    (14, 14): ([0, 20, 24, 42, 46],
               [0.2820947917738788, -0.1795148674924672, 0.1517177540482855,
                0.0711263801595843, 0.1881827135584983]),
    (15, 0): ([15], [0.2820947917738782]),
    (15, 1): ([4, 16, 18],
              [-0.2261790131595404, 0.2303294329808904, 0.0435281713775681]),
    (15, 2): ([23], [0.1628675039676400]),
    (15, 3): ([8, 22, 24],
              [0.2261790131595402, -0.0435281713775682, 0.2303294329808904]),
    (15, 4): ([1, 11, 25, 29],
              [-0.2261790131595403, 0.0940315972579595, 0.2455320005465368,
               -0.0169433177293593]),
    (15, 5): ([10, 26, 28],
              [-0.1486770096793975, 0.1552880720369528, 0.0448278050962364]),
    (15, 6): ([15, 33], [-0.2102610435016800, 0.1267921798770302]),
    (15, 7): ([14, 32, 34],
              [0.1486770096793976, -0.0448278050962365, 0.1552880720369529]),
    (15, 8): ([3, 13, 31, 35],
              [0.2261790131595401, -0.0940315972579593, 0.0169433177293592,
               0.2455320005465367]),
    (15, 9): ([36], [0.2548005986729752]),
    (15, 10): ([5, 19, 37, 41],
               [-0.1486770096793977, 0.0993225845992799, 0.1801712311720523,
                -0.0221775454765498]),
    (15, 11): ([4, 16, 18, 38, 40],
               [0.0940315972579595, -0.1175200669506001, -0.1332552305189783,
                0.1214714192760309, 0.0443550909531000]),
    (15, 12): ([23, 45], [-0.2035507268673356, 0.1086473403298334]),
    (15, 13): ([8, 22, 24, 44, 46],
               [-0.0940315972579593, 0.1332552305189782, -0.1175200669506001,
                -0.0443550909530999, 0.1214714192760309]),
    (15, 14): ([7, 21, 43, 47],
               [0.1486770096793976, -0.0993225845992798, 0.0221775454765500,
                0.1801712311720528]),
    (15, 15): ([0, 6, 20, 42, 48],
               [0.2820947917738780, -0.2102610435016802, 0.0769349432110579,
                -0.0118543966932641, 0.2548005986729751])}
