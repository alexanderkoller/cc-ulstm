import sys

with open("data/snli_1.0/cc/sent1/tokens.txt") as ftokens:
    with open("data/snli_1.0/cc/sent1/tags.txt") as ftags:
        with open("data/snli_1.0/cc/sent1/bconst_theta_0.9") as fbconst:
            with open("data/snli_1.0/cc/sent1/econst_theta_0.9") as feconst:
                for ltokens, ltags, lbconst, leconst in zip(ftokens, ftags, fbconst, feconst):
                    print("\n")
                    print(ltokens.strip())
                    print(lbconst.strip())
                    print(leconst.strip())
